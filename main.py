"""
main.py — Bells Generator
-------------------------
Additive synthesis bell sequencer in C minor.

Usage:
    python main.py                          # render default preset to WAV
    python main.py --list                   # list available presets
    python main.py --preset melodic_ascending
    python main.py --key "D minor" --bpm 72
    python main.py --gui                    # launch web UI
"""

import argparse
import sys
from pathlib import Path

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


def list_presets():
    from engine.presets import PRESETS
    print(f"{'ID':<22} {'Name':<26} BPM   Reverb  Decay")
    print("─" * 72)
    for pid, p in PRESETS.items():
        print(f"{pid:<22} {p['name']:<26} {p['default_bpm']:<6}"
              f"{p['default_reverb']:<8.2f}{p['default_decay']:.1f}×")
    print(f"\n{len(PRESETS)} preset(s).")


def render(preset_id: str, bpm: float, key: str,
           reverb_room: float, decay_mult: float, output: Path):
    from engine.synth import render_track, stereo_to_wav_bytes
    from engine.presets import PRESETS, KEYS

    preset = PRESETS.get(preset_id)
    if preset is None:
        sys.exit(f"Unknown preset '{preset_id}'. Use --list to see options.")

    key_semitones = KEYS.get(key, 0)
    all_events    = preset["melody"] + preset["bass"]

    print(f"🔔 Rendering '{preset['name']}' | "
          f"Key: {key} | BPM: {bpm} | Reverb: {reverb_room:.2f} | "
          f"Decay: {decay_mult:.1f}×")

    audio = render_track(
        all_events,
        total_beats   = preset["total_beats"],
        bpm           = bpm,
        reverb_room   = reverb_room,
        decay_mult    = decay_mult,
        key_semitones = key_semitones,
    )
    wav_bytes = stereo_to_wav_bytes(audio)
    output.write_bytes(wav_bytes)
    dur = len(audio) / 48_000
    print(f"  ✓ Saved: {output}  ({dur:.1f}s)")


def main():
    from engine.presets import PRESETS, KEYS

    parser = argparse.ArgumentParser(description="Bells Generator — Melodic bell synthesiser")
    parser.add_argument("--gui",     action="store_true", help="Launch web UI")
    parser.add_argument("--list",    action="store_true", help="List presets")
    parser.add_argument("--preset",  default="meditation",
                        choices=list(PRESETS.keys()), help="Preset to render")
    parser.add_argument("--key",     default=None,
                        choices=list(KEYS.keys()), help="Musical key (default: preset default)")
    parser.add_argument("--bpm",     type=float, default=None, help="Beats per minute")
    parser.add_argument("--reverb",  type=float, default=None, help="Reverb room size 0–1")
    parser.add_argument("--decay",   type=float, default=None, help="Bell decay multiplier")
    parser.add_argument("--output",  type=Path,  default=None, help="Output WAV path")
    args = parser.parse_args()

    if args.list:
        list_presets()
        return

    if args.gui:
        try:
            from engine.web_server import launch_gui
        except ImportError as e:
            sys.exit(f"Web UI requires fastapi and uvicorn.\nError: {e}")
        launch_gui()
        return

    preset = PRESETS[args.preset]
    bpm         = args.bpm    or preset["default_bpm"]
    key         = args.key    or "C minor"
    reverb_room = args.reverb if args.reverb is not None else preset["default_reverb"]
    decay_mult  = args.decay  if args.decay  is not None else preset["default_decay"]
    output      = args.output or EXPORT_DIR / f"{args.preset}.wav"

    render(args.preset, bpm, key, reverb_room, decay_mult, output)


if __name__ == "__main__":
    main()
