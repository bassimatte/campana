"""
engine/web_server.py
--------------------
FastAPI web UI for the Bells Generator.
Launch: python main.py --gui
"""

import asyncio
import io
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager, suppress
from pathlib import Path

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    from starlette.background import BackgroundTask
    import uvicorn
except ImportError:
    raise ImportError(
        "Web UI requires fastapi and uvicorn.\n"
        "Install with: pip install fastapi uvicorn[standard]"
    )

import numpy as np
from .synth import render_chunk, render_track, stereo_to_wav_bytes, generate_bell_events, _LOOKAHEAD_BEATS, note_freq, transpose_note
from .presets import PRESETS, KEYS, SCALES, resolve_preset_id

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

# ── Background render jobs (async export) ─────────────────────────────────────
def _positive_number_env(name: str, default: float) -> float:
    try:
        return max(1.0, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


_JOBS_MAX = int(_positive_number_env("CAMPANA_RENDER_JOBS_MAX", 20))
_RENDER_MAX_CONCURRENT = int(
    _positive_number_env("CAMPANA_RENDER_MAX_CONCURRENT", 2)
)
_RENDER_JOB_TTL_SECONDS = _positive_number_env("CAMPANA_RENDER_JOB_TTL", 600)
_RENDER_CLEANUP_INTERVAL_SECONDS = _positive_number_env(
    "CAMPANA_RENDER_CLEANUP_INTERVAL", 30
)

_render_jobs: dict = {}
_render_jobs_lock = threading.RLock()
_render_slots = threading.BoundedSemaphore(_RENDER_MAX_CONCURRENT)
_render_tempdir = tempfile.TemporaryDirectory(prefix="campana-renders-")
_RENDER_DIR = Path(_render_tempdir.name)


def _delete_render_file(path) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logging.warning("could not delete render file %s", path, exc_info=True)


def _trim_allocator_memory() -> None:
    """Ask glibc to return unused heap pages after a large render."""
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        malloc_trim = ctypes.CDLL(None).malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except (AttributeError, OSError):
        logging.debug("malloc_trim is not available on this platform")


def _job_paths(job: dict) -> set[Path]:
    return {
        Path(path)
        for path in (job.get("file_path"), job.get("work_path"))
        if path
    }


def _cleanup_render_jobs(*, now=None, force: bool = False) -> int:
    """Remove expired job records and their temporary files.

    Pending jobs expire from their creation time so abandoned long-running work
    cannot stay addressable forever. Terminal jobs expire from their most recent
    poll/download access. A worker whose pending record has expired notices that
    the record is gone and deletes any result it subsequently produces.
    """
    current = time.time() if now is None else now
    removed: list[dict] = []
    with _render_jobs_lock:
        for job_id, job in list(_render_jobs.items()):
            if job["status"] == "pending":
                age = current - job["created_at"]
            else:
                age = current - job["last_accessed_at"]
            if force or age >= _RENDER_JOB_TTL_SECONDS:
                removed.append(_render_jobs.pop(job_id))

    for job in removed:
        for path in _job_paths(job):
            _delete_render_file(path)
    if removed:
        logging.info("cleaned up %d expired Campana render job(s)", len(removed))
        _trim_allocator_memory()
    return len(removed)


def _register_render_job(job_id: str) -> bool:
    """Register a pending job, evicting the oldest terminal job if necessary."""
    now = time.time()
    evicted = None
    with _render_jobs_lock:
        if len(_render_jobs) >= _JOBS_MAX:
            terminal = [
                (candidate_id, job)
                for candidate_id, job in _render_jobs.items()
                if job["status"] != "pending"
            ]
            if not terminal:
                return False
            oldest_id, _ = min(
                terminal, key=lambda item: item[1]["created_at"]
            )
            evicted = _render_jobs.pop(oldest_id)

        _render_jobs[job_id] = {
            "status": "pending",
            "file_path": None,
            "work_path": None,
            "fname": None,
            "media_type": "audio/wav",
            "error": None,
            "created_at": now,
            "completed_at": None,
            "last_accessed_at": now,
        }

    if evicted:
        for path in _job_paths(evicted):
            _delete_render_file(path)
    return True


async def _render_cleanup_loop() -> None:
    while True:
        await asyncio.sleep(_RENDER_CLEANUP_INTERVAL_SECONDS)
        await asyncio.to_thread(_cleanup_render_jobs)


@asynccontextmanager
async def _app_lifespan(_app):
    cleanup_task = asyncio.create_task(_render_cleanup_loop())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        await asyncio.to_thread(_cleanup_render_jobs, force=True)


def _convert_audio_file(wav_path: Path, fmt: str) -> tuple:
    """Convert a WAV file incrementally and return (path, media type, ext).

    Conversion reads bounded blocks instead of loading the complete WAV and
    encoded result into memory at the same time. Missing optional encoders fall
    back to the already-rendered WAV file.
    """
    fmt = (fmt or "wav").lower().strip()
    if fmt == "wav":
        return wav_path, "audio/wav", "wav"
    try:
        import soundfile as sf
    except ImportError:
        logging.warning("soundfile not installed — falling back to WAV")
        return wav_path, "audio/wav", "wav"

    if fmt == "flac":
        output_path = wav_path.with_suffix(".flac")
        with sf.SoundFile(str(wav_path), "r") as source:
            with sf.SoundFile(
                str(output_path),
                "w",
                samplerate=source.samplerate,
                channels=source.channels,
                format="FLAC",
                subtype="PCM_16",
            ) as target:
                while True:
                    block = source.read(65_536, dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    target.write(block)
        _delete_render_file(wav_path)
        return output_path, "audio/flac", "flac"

    if fmt == "mp3":
        try:
            import lameenc
        except ImportError:
            logging.warning("lameenc not installed — falling back to WAV")
            return wav_path, "audio/wav", "wav"

        output_path = wav_path.with_suffix(".mp3")
        with sf.SoundFile(str(wav_path), "r") as source:
            enc = lameenc.Encoder()
            enc.set_bit_rate(320)
            enc.set_in_sample_rate(source.samplerate)
            enc.set_channels(source.channels)
            enc.set_quality(2)
            with output_path.open("wb") as target:
                while True:
                    block = source.read(65_536, dtype="float32", always_2d=True)
                    if not len(block):
                        break
                    audio_i16 = (
                        np.clip(block, -1.0, 1.0) * 32767
                    ).astype(np.int16)
                    target.write(enc.encode(audio_i16.flatten().tobytes()))
                target.write(enc.flush())
        _delete_render_file(wav_path)
        return output_path, "audio/mpeg", "mp3"

    return wav_path, "audio/wav", "wav"

app = FastAPI(title="Bells Generator", version="1.0", lifespan=_app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bassimatte.github.io",
        "https://campana-production.up.railway.app",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Events"],
)

_EPOCH_BEATS = 16.0   # generate notes in independent 16-beat windows → truly infinite


def _epoch_events(scale_mode: str, gen_params: dict, seed_base: int,
                  start_beat: float, end_beat: float) -> list:
    """Return events at absolute beat positions covering [start_beat, end_beat).
    Each 16-beat epoch has its own seed, so the music never repeats."""
    first_ep = max(0, int(start_beat // _EPOCH_BEATS))
    last_ep  = int(end_beat // _EPOCH_BEATS)
    events: list = []
    for ep in range(first_ep, last_ep + 1):
        ep_seed   = seed_base + ep * 997
        ep_events = generate_bell_events(scale_mode, _EPOCH_BEATS, gen_params, ep_seed)
        ep_start  = ep * _EPOCH_BEATS
        events.extend((e[0] + ep_start, e[1], e[2], e[3]) for e in ep_events)
    return events

def _parse_render_params(body: dict) -> dict:
    return dict(
        preset_id        = resolve_preset_id(body.get("preset", "sera")),
        bpm              = float(body.get("bpm", 50)),
        key              = body.get("key", "C minor"),
        reverb_room      = float(body.get("reverb_room", 0.5)),
        reverb_damping   = float(body.get("reverb_damping", 0.4)),
        reverb_width     = float(body.get("reverb_width", 0.8)),
        reverb_wet       = float(body.get("reverb_wet", 0.35)),
        decay_mult       = float(body.get("decay_mult", 1.4)),
        octave_spread    = float(body.get("octave_spread", 0.0)),
        base_octave_shift= int(body.get("base_octave_shift", 0)),
        humanize         = float(body.get("humanize", 0.0)),
        density          = float(body.get("density", 1.0)),
        texture          = str(body.get("texture", "tubular")),
        attack_ms        = float(body.get("attack_ms", 0.0)),
        beating          = float(body.get("beating", 0.0)),
        strike_level     = float(body.get("strike_level", 0.0)),
        delay_time       = float(body.get("delay_time", 300.0)),
        delay_feedback   = float(body.get("delay_feedback", 0.35)),
        delay_wet        = float(body.get("delay_wet", 0.0)),
        time_scatter     = float(body.get("time_scatter", 0.0)),
        shimmer          = float(body.get("shimmer", 0.0)),
        scale_mode       = str(body.get("scale_mode", "minor")),
        seed_base        = int(body.get("seed_base", 42)),
    )


def _do_preview(p: dict, body: dict, chunk_beats: float) -> bytes:
    preset        = PRESETS.get(p["preset_id"]) or next(iter(PRESETS.values()))
    key_semitones = KEYS.get(p["key"], 0)

    # Raw beat offset — no modulo. Time is infinite; epochs handle the variety.
    raw_offset = float(body.get("beat_offset", 0.0))

    # Collect events from all epochs overlapping [raw_offset - LOOK, raw_offset + chunk_beats]
    all_events = _epoch_events(
        p["scale_mode"],
        preset.get("gen_params", {}),
        p["seed_base"],
        start_beat = raw_offset - _LOOKAHEAD_BEATS,
        end_beat   = raw_offset + chunk_beats,
    )

    audio = render_chunk(
        all_events,
        beat_offset        = raw_offset,
        chunk_beats        = chunk_beats,
        total_beats        = 999_999,   # effectively infinite — no looping
        bpm                = p["bpm"],
        reverb_room        = p["reverb_room"],
        reverb_damping     = p["reverb_damping"],
        reverb_width       = p["reverb_width"],
        reverb_wet         = p["reverb_wet"],
        decay_mult         = p["decay_mult"],
        key_semitones      = key_semitones,
        octave_spread      = p["octave_spread"],
        base_octave_shift  = p["base_octave_shift"],
        humanize           = p["humanize"],
        density            = p["density"],
        texture            = p["texture"],
        attack_ms          = p["attack_ms"],
        beating            = p["beating"],
        strike_level       = p["strike_level"],
        seed               = p["seed_base"] + int(raw_offset) * 137,
        delay_time         = p["delay_time"],
        delay_feedback     = p["delay_feedback"],
        delay_wet          = p["delay_wet"],
        time_scatter       = p["time_scatter"],
        shimmer            = p["shimmer"],
        scale_mode         = p["scale_mode"],
    )

    wav = stereo_to_wav_bytes(audio)

    # Build note-event list for the visualiser (events in [0, chunk_beats) window only)
    beat_dur  = 60.0 / p["bpm"]
    key_shift = KEYS.get(p["key"], 0) + p["base_octave_shift"] * 12
    viz_evts  = []
    for b, n, d, v in all_events:
        rel = b - raw_offset
        if 0.0 <= rel < chunk_beats:
            try:
                t_note = transpose_note(n, key_shift)
                f      = round(note_freq(t_note), 1)
            except Exception:
                f = 440.0
            viz_evts.append({"t": round(rel * beat_dur, 3), "f": f, "v": round(float(v), 2)})

    return Response(content=wav, media_type="audio/wav",
                    headers={"X-Events": json.dumps(viz_evts)})


def _write_full_render(p: dict, buf) -> None:
    """Write an export to a binary file-like object in bounded-size chunks.

    The first chunk calibrates the amplitude scale; all subsequent chunks use
    the same scale so level is consistent throughout the file.
    """
    from .synth import SAMPLE_RATE, render_chunk as _rc

    preset        = PRESETS.get(p["preset_id"]) or next(iter(PRESETS.values()))
    key_semitones = KEYS.get(p["key"], 0)
    # export_beats set by the caller from user-selected duration;
    # fall back to preset default if called without it.
    export_beats  = float(p.get("export_beats") or preset["total_beats"])
    bpm           = p["bpm"]
    beat_dur      = 60.0 / bpm

    CHUNK_BEATS   = 32.0

    all_events = [e for e in _epoch_events(
        p["scale_mode"],
        preset.get("gen_params", {}),
        p["seed_base"],
        start_beat = 0,
        end_beat   = export_beats,
    ) if e[0] < export_beats]

    ckw = dict(
        total_beats       = 999_999,
        bpm               = bpm,
        reverb_room       = p["reverb_room"],
        reverb_damping    = p.get("reverb_damping", 0.4),
        reverb_width      = p.get("reverb_width", 0.8),
        reverb_wet        = p.get("reverb_wet", 0.35),
        decay_mult        = p["decay_mult"],
        key_semitones     = key_semitones,
        octave_spread     = p.get("octave_spread", 0.0),
        base_octave_shift = p.get("base_octave_shift", 0),
        humanize          = p.get("humanize", 0.0),
        density           = p.get("density", 1.0),
        texture           = p.get("texture", "tubular"),
        attack_ms         = p.get("attack_ms", 0.0),
        beating           = p.get("beating", 0.0),
        strike_level      = p.get("strike_level", 0.0),
        seed              = p["seed_base"],
        delay_time        = p.get("delay_time", 300.0),
        delay_feedback    = p.get("delay_feedback", 0.35),
        delay_wet         = p.get("delay_wet", 0.0),
        time_scatter      = p.get("time_scatter", 0.0),
        shimmer           = p.get("shimmer", 0.0),
        scale_mode        = p.get("scale_mode", "minor"),
    )

    # ── Pre-compute total WAV size for a correct header ────────────────────────
    tail_s       = 4.5 * max(p.get("decay_mult", 1.0), 1.0)
    total_secs   = export_beats * beat_dur + tail_s
    total_frames = int(total_secs * SAMPLE_RATE)
    CH, SW       = 2, 2       # stereo, 16-bit
    data_bytes   = total_frames * CH * SW

    # WAV header (44 bytes)
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_bytes))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<H", 1))           # PCM
    buf.write(struct.pack("<H", CH))
    buf.write(struct.pack("<I", SAMPLE_RATE))
    buf.write(struct.pack("<I", SAMPLE_RATE * CH * SW))
    buf.write(struct.pack("<H", CH * SW))
    buf.write(struct.pack("<H", SW * 8))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_bytes))

    scale         = None   # calibrated from first chunk, reused for all
    frames_left   = total_frames
    beat_offset   = 0.0

    while beat_offset < export_beats and frames_left > 0:
        this_chunk = min(CHUNK_BEATS, export_beats - beat_offset)
        is_last    = (beat_offset + this_chunk >= export_beats)

        audio = _rc(all_events, beat_offset, this_chunk, **ckw)

        if not is_last:
            # Keep only the music window; discard per-chunk reverb tail to
            # avoid a "reverb restart" hiccup at the next chunk boundary.
            music_samp = int(this_chunk * beat_dur * SAMPLE_RATE)
            audio      = audio[:music_samp]

        # Calibrate loudness from first chunk, clip subsequent chunks to that scale
        peak = float(np.max(np.abs(audio)))
        if scale is None:
            scale = (0.80 / peak) if peak > 1e-9 else 1.0

        pcm = (np.clip(audio * scale, -1.0, 1.0) * 32767).astype(np.int16)
        del audio

        frames_to_write = min(len(pcm), frames_left)
        buf.write(pcm[:frames_to_write].flatten(order="C").tobytes())
        frames_left -= frames_to_write
        del pcm

        beat_offset += this_chunk

    # Zero-pad if the last chunk was shorter than expected
    if frames_left > 0:
        buf.write(b"\x00" * frames_left * CH * SW)

def _do_full_render(p: dict) -> bytes:
    """Compatibility wrapper used by the command-line renderer."""
    buf = io.BytesIO()
    _write_full_render(p, buf)
    return buf.getvalue()


def _run_render_job(job_id: str, p: dict, fmt: str, name: str) -> None:
    """Render one export to disk and publish only its temporary file path."""
    wav_path = None
    result_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f"{job_id}-",
            suffix=".wav",
            dir=str(_RENDER_DIR),
            delete=False,
        ) as output:
            wav_path = Path(output.name)
            abandoned = False
            with _render_jobs_lock:
                job = _render_jobs.get(job_id)
                if job is None:
                    abandoned = True
                else:
                    job["work_path"] = str(wav_path)
            if not abandoned:
                _write_full_render(p, output)

        if abandoned:
            _delete_render_file(wav_path)
            return

        # The periodic cleanup may have expired this job while it rendered.
        # Avoid spending more CPU and disk space converting an orphaned WAV.
        with _render_jobs_lock:
            if job_id not in _render_jobs:
                _delete_render_file(wav_path)
                return

        result_path, media_type, ext = _convert_audio_file(wav_path, fmt)
        completed_at = time.time()
        with _render_jobs_lock:
            job = _render_jobs.get(job_id)
            if job is None:
                _delete_render_file(result_path)
                return
            job.update(
                file_path=str(result_path),
                work_path=None,
                fname=name.replace(" ", "_") + "." + ext,
                media_type=media_type,
                status="done",
                completed_at=completed_at,
                last_accessed_at=completed_at,
            )
    except Exception:
        tb = traceback.format_exc()
        logging.error("render job %s error:\n%s", job_id, tb)
        completed_at = time.time()
        with _render_jobs_lock:
            job = _render_jobs.get(job_id)
            if job is not None:
                job.update(
                    status="error",
                    error=tb,
                    work_path=None,
                    completed_at=completed_at,
                    last_accessed_at=completed_at,
                )
        for candidate in {
            path
            for path in (
                wav_path,
                result_path,
                wav_path.with_suffix(".flac") if wav_path else None,
                wav_path.with_suffix(".mp3") if wav_path else None,
            )
            if path
        }:
            _delete_render_file(candidate)
    finally:
        _render_slots.release()
        _trim_allocator_memory()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/favicon.png", include_in_schema=False)
def favicon():
    return FileResponse(
        _STATIC_DIR / "favicon.png",
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/")
def index():
    return FileResponse(
        _STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@app.get("/api/version")
async def get_version():
    """Return the running git commit SHA and GitHub repo URL."""
    sha = os.environ.get("GIT_SHA", "")
    if not sha:
        try:
            sha = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(Path(__file__).parent.parent),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            sha = "unknown"
    return JSONResponse({"sha": sha, "repo": "https://github.com/bassimatte/campana"})


@app.get("/api/presets")
def list_presets():
    keys = [
        "name", "description", "default_bpm", "default_key", "default_scale_mode",
        "default_reverb", "default_decay", "default_texture",
        "default_reverb_damping", "default_reverb_width", "default_reverb_wet",
        "default_delay_time", "default_delay_feedback", "default_delay_wet",
        "default_beating", "default_strike", "default_attack_ms",
        "default_humanize", "default_density", "default_octave_spread",
        "default_base_octave", "default_time_scatter", "default_shimmer",
    ]
    return JSONResponse([
        {"id": k, **{f: v[f] for f in keys if f in v}}
        for k, v in PRESETS.items()
    ])


@app.get("/api/scales")
def list_scales():
    return JSONResponse([{"id": k, "name": v} for k, v in SCALES.items()])


@app.get("/api/keys")
def list_keys():
    return JSONResponse(list(KEYS.keys()))


@app.post("/api/preview")
async def preview_audio(request: Request):
    body        = await request.json()
    p           = _parse_render_params(body)
    chunk_beats = float(body.get("chunk_beats", 12))
    try:
        return await asyncio.to_thread(_do_preview, p, body, chunk_beats)
    except Exception:
        tb = traceback.format_exc()
        logging.error("preview error:\n%s\nparams: %s", tb, p)
        return JSONResponse({"detail": tb}, status_code=500)


@app.post("/api/render")
async def start_render(request: Request):
    """Start a background render job. Returns {job_id} immediately."""
    _cleanup_render_jobs()
    body    = await request.json()
    p       = _parse_render_params(body)
    minutes = float(body.get("export_minutes", 5))
    fmt     = (body.get("export_format") or "wav").lower()
    p["export_beats"] = minutes * p["bpm"]   # convert user minutes → beats
    name    = (PRESETS.get(p["preset_id"]) or next(iter(PRESETS.values())))["name"]

    if not _render_slots.acquire(blocking=False):
        return JSONResponse(
            {"detail": "render capacity reached; retry shortly"},
            status_code=429,
            headers={"Retry-After": "30"},
        )

    job_id = uuid.uuid4().hex[:10]
    if not _register_render_job(job_id):
        _render_slots.release()
        return JSONResponse(
            {"detail": "render queue is full; retry shortly"},
            status_code=429,
            headers={"Retry-After": "30"},
        )

    try:
        threading.Thread(
            target=_run_render_job,
            args=(job_id, p, fmt, name),
            daemon=True,
        ).start()
    except Exception:
        with _render_jobs_lock:
            _render_jobs.pop(job_id, None)
        _render_slots.release()
        raise
    return JSONResponse({"job_id": job_id})


@app.get("/api/render/{job_id}")
async def render_status(job_id: str):
    """Poll render job status: {status: pending|done|error, error?: str}"""
    with _render_jobs_lock:
        job = _render_jobs.get(job_id)
        if not job:
            return JSONResponse({"status": "not_found"}, status_code=404)
        job["last_accessed_at"] = time.time()
        resp = {"status": job["status"]}
        if job["status"] == "error":
            resp["error"] = job.get("error", "unknown error")
    return JSONResponse(resp)


@app.get("/api/render/{job_id}/file")
async def download_render(job_id: str):
    """Stream the finished file and delete it when transmission completes."""
    with _render_jobs_lock:
        job = _render_jobs.get(job_id)
        if not job:
            return JSONResponse({"detail": "job not found"}, status_code=404)
        job["last_accessed_at"] = time.time()
        if job["status"] != "done" or not job["file_path"]:
            return JSONResponse({"detail": "not ready"}, status_code=202)
        file_path = Path(job["file_path"])
        fname = job["fname"]
        media_type = job.get("media_type", "audio/wav")
        _render_jobs.pop(job_id)

    if not file_path.is_file():
        return JSONResponse({"detail": "render file missing"}, status_code=404)
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
        background=BackgroundTask(_delete_render_file, file_path),
    )

# ── Launch ────────────────────────────────────────────────────────────────────

def launch_gui(host: str = "127.0.0.1", port: int = 8081):
    import webbrowser
    print(f"Bells Generator -- http://{host}:{port}")
    print("   Press Ctrl+C to stop.\n")

    def _open():
        import time; time.sleep(1.2)
        webbrowser.open(f"http://{host}:{port}")

    threading.Thread(target=_open, daemon=True).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
