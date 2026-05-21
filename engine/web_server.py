"""
engine/web_server.py
--------------------
FastAPI web UI for the Bells Generator.
Launch: python main.py --gui
"""

import asyncio
import io
import logging
import threading
import traceback
from pathlib import Path

try:
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
except ImportError:
    raise ImportError(
        "Web UI requires fastapi and uvicorn.\n"
        "Install with: pip install fastapi uvicorn[standard]"
    )

from .synth import render_chunk, render_track, stereo_to_wav_bytes
from .presets import PRESETS, KEYS, SCALES

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Bells Generator", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bassimatte.github.io",   # GitHub Pages
        "http://localhost:8081",           # local dev
        "http://127.0.0.1:8081",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_render_params(body: dict) -> dict:
    return dict(
        preset_id        = body.get("preset", "meditation"),
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
    )


def _do_preview(p: dict, body: dict, chunk_beats: float) -> bytes:
    preset        = PRESETS.get(p["preset_id"]) or next(iter(PRESETS.values()))
    key_semitones = KEYS.get(p["key"], 0)
    all_events    = preset["melody"] + preset["bass"]
    total_beats   = float(preset["total_beats"])
    beat_offset   = float(body.get("beat_offset", 0.0)) % total_beats

    audio = render_chunk(
        all_events,
        beat_offset        = beat_offset,
        chunk_beats        = chunk_beats,
        total_beats        = total_beats,
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
        seed               = 42 + int(beat_offset) * 137,
        delay_time         = p["delay_time"],
        delay_feedback     = p["delay_feedback"],
        delay_wet          = p["delay_wet"],
        time_scatter       = p["time_scatter"],
        shimmer            = p["shimmer"],
        scale_mode         = p["scale_mode"],
    )
    return stereo_to_wav_bytes(audio)


def _do_full_render(p: dict) -> bytes:
    preset        = PRESETS.get(p["preset_id"]) or next(iter(PRESETS.values()))
    key_semitones = KEYS.get(p["key"], 0)
    all_events    = preset["melody"] + preset["bass"]
    audio = render_track(
        all_events,
        total_beats        = preset["total_beats"],
        bpm                = p["bpm"],
        reverb_room        = p["reverb_room"],
        reverb_damping     = p.get("reverb_damping", 0.4),
        reverb_width       = p.get("reverb_width", 0.8),
        reverb_wet         = p.get("reverb_wet", 0.35),
        decay_mult         = p["decay_mult"],
        key_semitones      = key_semitones,
        octave_spread      = p.get("octave_spread", 0.0),
        base_octave_shift  = p.get("base_octave_shift", 0),
        humanize           = p.get("humanize", 0.0),
        density            = p.get("density", 1.0),
        texture            = p.get("texture", "tubular"),
        attack_ms          = p.get("attack_ms", 0.0),
        beating            = p.get("beating", 0.0),
        strike_level       = p.get("strike_level", 0.0),
        delay_time         = p.get("delay_time", 300.0),
        delay_feedback     = p.get("delay_feedback", 0.35),
        delay_wet          = p.get("delay_wet", 0.0),
        time_scatter       = p.get("time_scatter", 0.0),
        shimmer            = p.get("shimmer", 0.0),
        scale_mode         = p.get("scale_mode", "minor"),
    )
    return stereo_to_wav_bytes(audio)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(
        _STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


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
        wav = await asyncio.to_thread(_do_preview, p, body, chunk_beats)
    except Exception:
        tb = traceback.format_exc()
        logging.error("preview error:\n%s\nparams: %s", tb, p)
        return JSONResponse({"detail": tb}, status_code=500)
    return Response(content=wav, media_type="audio/wav")


@app.post("/api/render")
async def render_audio(request: Request):
    body  = await request.json()
    p     = _parse_render_params(body)
    name  = (PRESETS.get(p["preset_id"]) or next(iter(PRESETS.values())))["name"]
    fname = name.replace(" ", "_") + ".wav"
    try:
        wav = await asyncio.to_thread(_do_full_render, p)
    except Exception:
        tb = traceback.format_exc()
        logging.error("render error:\n%s\nparams: %s", tb, p)
        return JSONResponse({"detail": tb}, status_code=500)
    return StreamingResponse(
        io.BytesIO(wav),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
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
