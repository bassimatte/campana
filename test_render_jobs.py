import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine import web_server


class _JsonRequest:
    async def json(self):
        return {
            "preset": "sera",
            "bpm": 45,
            "export_minutes": 1,
            "export_format": "wav",
        }


class RenderJobRetentionTests(unittest.TestCase):
    def tearDown(self):
        web_server._cleanup_render_jobs(force=True)

    def test_expired_jobs_and_files_are_removed_without_new_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            expired_done = directory / "expired.wav"
            expired_pending = directory / "pending.wav"
            active = directory / "active.wav"
            for path in (expired_done, expired_pending, active):
                path.write_bytes(b"audio")

            with web_server._render_jobs_lock:
                web_server._render_jobs.update(
                    {
                        "expired-done": {
                            "status": "done",
                            "file_path": str(expired_done),
                            "work_path": None,
                            "created_at": 0.0,
                            "last_accessed_at": 0.0,
                        },
                        "expired-pending": {
                            "status": "pending",
                            "file_path": None,
                            "work_path": str(expired_pending),
                            "created_at": 0.0,
                            "last_accessed_at": 999.0,
                        },
                        "active": {
                            "status": "done",
                            "file_path": str(active),
                            "work_path": None,
                            "created_at": 900.0,
                            "last_accessed_at": 999.0,
                        },
                    }
                )

            with mock.patch.object(web_server, "_RENDER_JOB_TTL_SECONDS", 10):
                removed = web_server._cleanup_render_jobs(now=1000.0)

            self.assertEqual(removed, 2)
            self.assertFalse(expired_done.exists())
            self.assertFalse(expired_pending.exists())
            self.assertTrue(active.exists())
            self.assertEqual(set(web_server._render_jobs), {"active"})

    def test_periodic_cleanup_runs_while_no_requests_arrive(self):
        with tempfile.TemporaryDirectory() as directory:
            expired_file = Path(directory) / "periodic.wav"
            expired_file.write_bytes(b"audio")
            with web_server._render_jobs_lock:
                web_server._render_jobs["periodic"] = {
                    "status": "done",
                    "file_path": str(expired_file),
                    "work_path": None,
                    "created_at": 0.0,
                    "last_accessed_at": 0.0,
                }

            async def wait_for_cleanup():
                task = asyncio.create_task(web_server._render_cleanup_loop())
                try:
                    await asyncio.sleep(0.05)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

            with mock.patch.object(
                web_server, "_RENDER_CLEANUP_INTERVAL_SECONDS", 0.01
            ), mock.patch.object(web_server, "_RENDER_JOB_TTL_SECONDS", 1):
                asyncio.run(wait_for_cleanup())

            self.assertNotIn("periodic", web_server._render_jobs)
            self.assertFalse(expired_file.exists())

    def test_job_limit_evicts_terminal_jobs_but_not_running_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            completed_file = Path(directory) / "completed.wav"
            completed_file.write_bytes(b"audio")

            with mock.patch.object(web_server, "_JOBS_MAX", 2):
                self.assertTrue(web_server._register_render_job("completed"))
                with web_server._render_jobs_lock:
                    web_server._render_jobs["completed"].update(
                        status="done", file_path=str(completed_file)
                    )
                self.assertTrue(web_server._register_render_job("pending"))
                self.assertTrue(web_server._register_render_job("new"))
                self.assertFalse(completed_file.exists())
                self.assertEqual(set(web_server._render_jobs), {"pending", "new"})
                self.assertFalse(web_server._register_render_job("overflow"))

    def test_completed_render_retains_only_a_temporary_file_path(self):
        self.assertTrue(web_server._register_render_job("file-job"))
        self.assertTrue(web_server._render_slots.acquire(blocking=False))

        def write_small_render(_params, output):
            output.write(b"RIFF-test-audio")

        with mock.patch.object(
            web_server, "_write_full_render", side_effect=write_small_render
        ):
            web_server._run_render_job("file-job", {}, "wav", "Test Render")

        with web_server._render_jobs_lock:
            job = dict(web_server._render_jobs["file-job"])
        self.assertEqual(job["status"], "done")
        self.assertNotIn("data", job)
        self.assertIsNotNone(job["completed_at"])
        self.assertIsNone(job["work_path"])
        self.assertTrue(Path(job["file_path"]).is_file())

    def test_job_expired_during_render_skips_conversion_and_deletes_wav(self):
        files_before = set(web_server._RENDER_DIR.iterdir())
        self.assertTrue(web_server._register_render_job("expired-mid-render"))
        self.assertTrue(web_server._render_slots.acquire(blocking=False))

        def write_then_expire(_params, output):
            output.write(b"RIFF-test-audio")
            with web_server._render_jobs_lock:
                web_server._render_jobs.pop("expired-mid-render")

        with mock.patch.object(
            web_server, "_write_full_render", side_effect=write_then_expire
        ), mock.patch.object(web_server, "_convert_audio_file") as convert:
            web_server._run_render_job(
                "expired-mid-render", {}, "mp3", "Expired Render"
            )

        convert.assert_not_called()
        self.assertEqual(set(web_server._RENDER_DIR.iterdir()), files_before)

    def test_download_removes_job_then_deletes_file_after_response(self):
        with tempfile.TemporaryDirectory() as directory:
            render_file = Path(directory) / "download.wav"
            render_file.write_bytes(b"RIFF-test-audio")
            self.assertTrue(web_server._register_render_job("download-job"))
            with web_server._render_jobs_lock:
                web_server._render_jobs["download-job"].update(
                    status="done",
                    file_path=str(render_file),
                    fname="Campana.wav",
                    completed_at=1.0,
                )

            response = asyncio.run(web_server.download_render("download-job"))
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("download-job", web_server._render_jobs)
            self.assertTrue(render_file.exists())
            asyncio.run(response.background())
            self.assertFalse(render_file.exists())

    def test_excess_render_work_gets_retryable_429(self):
        acquired = 0
        try:
            for _ in range(web_server._RENDER_MAX_CONCURRENT):
                self.assertTrue(web_server._render_slots.acquire(blocking=False))
                acquired += 1
            response = asyncio.run(web_server.start_render(_JsonRequest()))
            self.assertEqual(response.status_code, 429)
            self.assertEqual(response.headers["retry-after"], "30")
            self.assertIn(b"render capacity reached", response.body)
        finally:
            for _ in range(acquired):
                web_server._render_slots.release()


if __name__ == "__main__":
    unittest.main()
