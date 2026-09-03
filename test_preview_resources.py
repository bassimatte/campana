import asyncio
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from fastapi.responses import Response

from engine import web_server


ROOT = Path(__file__).parent


class _JsonRequest:
    def __init__(self, seed):
        self.seed = seed

    async def json(self):
        return {
            "preset": "festa",
            "bpm": 104,
            "seed_base": self.seed,
            "chunk_beats": 6,
        }


class PreviewResourceTests(unittest.TestCase):
    def test_preview_synthesis_is_serialized(self):
        active = 0
        max_active = 0
        state_lock = threading.Lock()

        def observed_render(_params, _body, _beats):
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            try:
                time.sleep(0.03)
                return Response(content=b"RIFF-test", media_type="audio/wav")
            finally:
                with state_lock:
                    active -= 1

        async def run_two_previews():
            return await asyncio.gather(
                web_server.preview_audio(_JsonRequest(1)),
                web_server.preview_audio(_JsonRequest(2)),
            )

        with mock.patch.object(
            web_server, "_preview_slots", threading.BoundedSemaphore(1)
        ), mock.patch.object(
            web_server, "_do_preview_with_cleanup", side_effect=observed_render
        ):
            responses = asyncio.run(run_two_previews())

        self.assertEqual(max_active, 1)
        self.assertEqual([response.status_code for response in responses], [200, 200])

    def test_allocator_cleanup_runs_when_preview_fails(self):
        with mock.patch.object(
            web_server, "_do_preview", side_effect=RuntimeError("render failed")
        ), mock.patch.object(web_server, "_trim_allocator_memory") as trim:
            with self.assertRaisesRegex(RuntimeError, "render failed"):
                web_server._do_preview_with_cleanup({}, {}, 6)

        trim.assert_called_once_with()

    def test_both_frontends_retry_transient_preview_network_errors(self):
        for relative_path in ("docs/index.html", "engine/static/index.html"):
            source = (ROOT / relative_path).read_text()
            self.assertIn("for (let attempt = 0; attempt < 3; attempt++)", source)
            self.assertIn("if (attempt === 2) throw err", source)


if __name__ == "__main__":
    unittest.main()
