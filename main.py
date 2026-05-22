"""
main.py — Campana
-----------------
Generative bell synthesizer.

Usage:
    python main.py --list                          # list presets
    python main.py                                 # export default preset (sera, 2 min)
    python main.py --preset tempio                 # export one preset
    python main.py --preset sera --minutes 5       # 5-minute export
    python main.py --all                           # export all presets
    python main.py --all --minutes 2               # all presets, 2 min each
    python main.py --gui                           # launch web UI
"""

import argparse
import sys
from pathlib import Path

EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)


def list_presets():
    from engine.presets import PRESETS
    print(f"{'ID':<16} {'Name':<16} BPM   Key        Scale")
    print("─" * 60)
    for pid, p in PRESETS.items():
        print(f"{pid:<16} {p['name']:<16} {p['default_bpm']:<6}"
              f"{p.get('default_key','C'):<11}{p.get('default_scale_mode','minor')}")
    print(f"\n{len(PRESETS)} preset(s).")


def render_preset(preset_id: str, minutes: float, output: Path):
    from engine.web_server import _do_full_render, _parse_render_params
    from engine.presets import PRESETS

    preset = PRESETS.get(preset_id)
    if preset is None:
        sys.exit(f"Unknown preset '{preset_id}'. Use --list to see options.")

    p = _parse_render_params({
        "preset":     preset_id,
        "bpm":        preset["default_bpm"],
        "key":        preset.get("default_key", "C"),
        "reverb_room":      preset.get("default_reverb", 0.75),
        "reverb_damping":   preset.get("default_reverb_damping", 0.4),
        "reverb_width":     preset.get("default_reverb_width", 0.8),
        "reverb_wet":       preset.get("default_reverb_wet", 0.35),
        "decay_mult":       preset.get("default_decay", 1.4),
        "texture":          preset.get("default_texture", "tubular"),
        "beating":          preset.get("default_beating", 0.0),
        "strike_level":     preset.get("default_strike", 0.0),
        "attack_ms":        preset.get("default_attack_ms", 0.0),
        "humanize":         preset.get("default_humanize", 0.0),
        "density":          preset.get("default_density", 1.0),
        "octave_spread":    preset.get("default_octave_spread", 0.0),
        "base_octave_shift":preset.get("default_base_octave", 0),
        "time_scatter":     preset.get("default_time_scatter", 0.0),
        "delay_time":       preset.get("default_delay_time", 300),
        "delay_feedback":   preset.get("default_delay_feedback", 0.35),
        "delay_wet":        preset.get("default_delay_wet", 0.0),
        "shimmer":          preset.get("default_shimmer", 0.0),
        "scale_mode":       preset.get("default_scale_mode", "minor"),
        "seed_base":        42,
    })
    p["export_beats"] = minutes * p["bpm"]

    print(f"  🔔 {preset['name']:<14} {minutes:.0f} min @ {p['bpm']} BPM …", end="", flush=True)
    wav = _do_full_render(p)
    output.write_bytes(wav)
    dur = len(wav) / (48_000 * 2 * 2)  # bytes / (sr * channels * bytes_per_sample)
    print(f"  {dur/60:.1f} min  →  {output}  ({len(wav)/1e6:.1f} MB)")


def main():
    from engine.presets import PRESETS

    parser = argparse.ArgumentParser(description="Campana — generative bell synthesizer")
    parser.add_argument("--gui",     action="store_true", help="Launch web UI")
    parser.add_argument("--list",    action="store_true", help="List presets")
    parser.add_argument("--all",     action="store_true", help="Export all presets")
    parser.add_argument("--preset",  default="sera",
                        choices=list(PRESETS.keys()), help="Preset ID (default: sera)")
    parser.add_argument("--minutes", type=float, default=2.0,
                        help="Export duration in minutes (default: 2)")
    parser.add_argument("--output",  type=Path, default=None,
                        help="Output WAV path (single preset only)")
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

    if args.all:
        print(f"Exporting all {len(PRESETS)} presets ({args.minutes:.0f} min each) → {EXPORT_DIR}/\n")
        for pid in PRESETS:
            render_preset(pid, args.minutes, EXPORT_DIR / f"{pid}.wav")
        print("\nDone.")
    else:
        output = args.output or EXPORT_DIR / f"{args.preset}.wav"
        render_preset(args.preset, args.minutes, output)


if __name__ == "__main__":
    main()
