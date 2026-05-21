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

from .synth import render_chunk, render_track, stereo_to_wav_bytes, generate_bell_events, _LOOKAHEAD_BEATS, note_freq, transpose_note
from .presets import PRESETS, KEYS, SCALES

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Bells Generator", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://bassimatte.github.io",
        "http://localhost:8081",
        "http://127.0.0.1:8081",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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


def _do_full_render(p: dict) -> bytes:
    preset        = PRESETS.get(p["preset_id"]) or next(iter(PRESETS.values()))
    key_semitones = KEYS.get(p["key"], 0)
    export_beats  = float(preset["total_beats"])   # use preset duration for export length
    all_events    = [e for e in _epoch_events(
        p["scale_mode"],
        preset.get("gen_params", {}),
        p["seed_base"],
        start_beat = 0,
        end_beat   = export_beats,
    ) if e[0] < export_beats]
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
        return await asyncio.to_thread(_do_preview, p, body, chunk_beats)
    except Exception:
        tb = traceback.format_exc()
        logging.error("preview error:\n%s\nparams: %s", tb, p)
        return JSONResponse({"detail": tb}, status_code=500)


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
