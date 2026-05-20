"""
Melodic Bells Generator — C minor, 60 BPM
Synthesises bell tones using additive synthesis with inharmonic partials
and a simple Schroeder reverb. Outputs a stereo 48 kHz WAV file.
"""

import numpy as np
import wave
import struct

# ── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 48_000
BPM         = 60
BEAT_DUR    = 60.0 / BPM          # 1.0 second per beat
OUTPUT_FILE = "bells_generated.wav"

# ── Note frequency table (C minor scale, octaves 3–6) ────────────────────────
def note_freq(name: str) -> float:
    """Return frequency in Hz for a note name like 'Eb5' or 'Bb4'."""
    names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    aliases = {'C#': 'Db', 'D#': 'Eb', 'E#': 'F', 'Gb': 'F#', 'G#': 'Ab', 'A#': 'Bb', 'B#': 'C'}
    octave = int(name[-1])
    note   = name[:-1]
    note   = aliases.get(note, note)
    semitone = names.index(note)
    # MIDI-style: C4 = 261.63 Hz = A4-9 semitones, A4 = 440 Hz
    midi = (octave + 1) * 12 + semitone
    return 440.0 * 2 ** ((midi - 69) / 12)

# ── Bell tone synthesis ───────────────────────────────────────────────────────
# Inharmonic partials adapted from studies of struck metallic bells.
# Each entry: (frequency_ratio, relative_amplitude, decay_time_seconds)
BELL_PARTIALS = [
    (1.000,  1.00,  2.8),   # fundamental  — longest ring
    (2.756,  0.55,  1.4),   # minor-tenth partial
    (5.404,  0.25,  0.7),   # upper mid
    (8.933,  0.12,  0.35),  # high shimmer
    (13.34,  0.06,  0.18),  # very high attack click
]

def bell_tone(freq: float, duration: float, velocity: float = 1.0) -> np.ndarray:
    """Return mono bell tone samples (float64, peak ≤ 1.0)."""
    n   = int(SAMPLE_RATE * duration)
    t   = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros(n, dtype=np.float64)
    for ratio, amp, tau in BELL_PARTIALS:
        env  = np.exp(-t / tau)
        sig += amp * np.sin(2 * np.pi * freq * ratio * t) * env
    # Normalise to [-1, 1] then scale by velocity
    peak = np.max(np.abs(sig)) or 1.0
    return sig / peak * velocity

# ── Reverb (Schroeder network: 4 comb + 2 allpass) ───────────────────────────
def schroeder_reverb(signal: np.ndarray, room_scale: float = 0.5,
                     wet: float = 0.35) -> np.ndarray:
    """Simple Schroeder reverb — returns the wet+dry mix."""
    sr = SAMPLE_RATE

    def comb(x, delay_ms, gain):
        d   = int(delay_ms * sr / 1000)
        buf = np.zeros(d)
        out = np.zeros(len(x))
        for i, s in enumerate(x):
            idx       = i % d
            out[i]    = buf[idx]
            buf[idx]  = s + gain * buf[idx]
        return out

    def allpass(x, delay_ms, gain=0.5):
        d   = int(delay_ms * sr / 1000)
        buf = np.zeros(d)
        out = np.zeros(len(x))
        for i, s in enumerate(x):
            idx      = i % d
            b        = buf[idx]
            out[i]   = -gain * s + b
            buf[idx] = s + gain * b
        return out

    g = 0.76 + 0.14 * room_scale          # feedback gain
    combs = [
        comb(signal, 29.7, g),
        comb(signal, 37.1, g * 0.98),
        comb(signal, 41.1, g * 0.96),
        comb(signal, 43.7, g * 0.94),
    ]
    wet_sig = sum(combs) / len(combs)
    wet_sig = allpass(wet_sig, 5.0)
    wet_sig = allpass(wet_sig, 1.7)
    return signal * (1 - wet) + wet_sig * wet

# ── Melody definition ─────────────────────────────────────────────────────────
# Each event: (beat_start, note_name, duration_beats, velocity 0‥1)
# C natural minor scale: C D Eb F G Ab Bb
# Pattern spans ~56 beats (≈ 56 s at 60 BPM) in four 14-beat phrases.

MELODY = [
    # ── Phrase 1: ascending C minor arpeggio / melodic line ──
    (0.0,  'Eb5', 0.75, 0.9),
    (0.75, 'G5',  0.75, 0.85),
    (1.5,  'Bb5', 0.75, 0.9),
    (2.25, 'C6',  1.5,  0.85),
    (3.75, 'Bb5', 0.75, 0.8),
    (4.5,  'G5',  0.75, 0.8),
    (5.25, 'Eb5', 0.75, 0.85),
    (6.0,  'C5',  2.0,  0.9),

    (8.0,  'F5',  0.75, 0.85),
    (8.75, 'Ab5', 0.75, 0.9),
    (9.5,  'Bb5', 0.75, 0.85),
    (10.25,'G5',  0.75, 0.8),
    (11.0, 'F5',  0.5,  0.75),
    (11.5, 'Eb5', 0.5,  0.8),
    (12.0, 'D5',  0.75, 0.85),
    (12.75,'Eb5', 1.25, 0.9),

    # ── Phrase 2: descending with chromatic colour ──
    (14.0, 'Bb5', 0.5,  0.88),
    (14.5, 'Ab5', 0.5,  0.85),
    (15.0, 'G5',  0.75, 0.9),
    (15.75,'F5',  0.5,  0.8),
    (16.25,'Eb5', 0.5,  0.85),
    (16.75,'D5',  0.5,  0.8),
    (17.25,'C5',  2.75, 0.9),

    (20.0, 'G5',  0.5,  0.8),
    (20.5, 'Bb5', 0.5,  0.85),
    (21.0, 'C6',  0.75, 0.9),
    (21.75,'Bb5', 0.5,  0.82),
    (22.25,'Ab5', 0.5,  0.8),
    (22.75,'G5',  1.25, 0.88),
    (24.0, 'F5',  0.5,  0.8),
    (24.5, 'Eb5', 0.5,  0.85),
    (25.0, 'D5',  0.5,  0.8),
    (25.5, 'Eb5', 2.5,  0.9),

    # ── Phrase 3: mid-range with bass tones ──
    (28.0, 'C5',  0.5,  0.85),
    (28.5, 'Eb5', 0.5,  0.88),
    (29.0, 'G5',  0.75, 0.9),
    (29.75,'F5',  0.5,  0.8),
    (30.25,'Eb5', 0.5,  0.85),
    (30.75,'C5',  1.25, 0.88),
    (32.0, 'D5',  0.5,  0.82),
    (32.5, 'F5',  0.75, 0.85),
    (33.25,'Ab5', 0.75, 0.88),
    (34.0, 'G5',  0.75, 0.9),
    (34.75,'F5',  0.5,  0.82),
    (35.25,'Eb5', 0.5,  0.85),
    (35.75,'D5',  2.25, 0.88),

    # ── Phrase 4: return to home / closing ──
    (38.0, 'Eb5', 0.75, 0.9),
    (38.75,'G5',  0.75, 0.88),
    (39.5, 'Bb5', 1.0,  0.9),
    (40.5, 'Ab5', 0.5,  0.82),
    (41.0, 'G5',  0.5,  0.85),
    (41.5, 'F5',  0.5,  0.8),
    (42.0, 'Eb5', 0.5,  0.88),
    (42.5, 'D5',  0.5,  0.82),
    (43.0, 'C5',  1.0,  0.9),
    (44.0, 'Eb5', 0.5,  0.85),
    (44.5, 'G5',  0.5,  0.88),
    (45.0, 'Bb5', 0.5,  0.9),
    (45.5, 'C6',  0.5,  0.88),
    (46.0, 'Bb5', 0.5,  0.85),
    (46.5, 'G5',  0.5,  0.82),
    (47.0, 'Eb5', 0.5,  0.88),
    (47.5, 'C5',  4.5,  0.9),   # long final note
]

# Bass / low accompaniment notes (sparser)
BASS = [
    (0.0,  'C3',  3.5,  0.7),
    (4.0,  'Bb2', 3.5,  0.65),
    (8.0,  'F3',  3.5,  0.7),
    (12.0, 'G3',  2.0,  0.65),
    (14.0, 'Eb3', 3.5,  0.7),
    (18.0, 'Bb2', 3.5,  0.65),
    (20.0, 'C3',  3.5,  0.7),
    (24.0, 'F3',  3.5,  0.65),
    (28.0, 'C3',  3.5,  0.7),
    (32.0, 'Ab2', 3.5,  0.65),
    (35.0, 'Bb2', 3.0,  0.65),
    (38.0, 'Eb3', 3.5,  0.7),
    (42.0, 'Bb2', 3.5,  0.65),
    (46.0, 'C3',  6.0,  0.7),
]

# ── Render ────────────────────────────────────────────────────────────────────
def render_track(events: list, total_beats: float,
                 reverb_room: float = 0.5, pan_spread: float = 0.4) -> np.ndarray:
    """Render all events into a stereo float buffer."""
    total_samples = int(total_beats * BEAT_DUR * SAMPLE_RATE) + SAMPLE_RATE * 5
    stereo = np.zeros((total_samples, 2), dtype=np.float64)

    rng = np.random.default_rng(42)   # deterministic slight pan jitter

    for beat, note, dur_beats, vel in events:
        freq        = note_freq(note)
        dur_samp    = dur_beats * BEAT_DUR
        dur_samp    = max(dur_samp, 0.05) + 3.0   # allow tail to ring
        start_samp  = int(beat * BEAT_DUR * SAMPLE_RATE)
        tone        = bell_tone(freq, dur_samp, velocity=vel)

        # Apply reverb to each note individually for realistic per-note tail
        tone = schroeder_reverb(tone, room_scale=reverb_room, wet=0.32)

        # Pan: higher notes slightly right, lower notes slightly left + small jitter
        base_pan = (note_freq(note) - 400) / 1200   # −1 … +1 range approx
        base_pan = np.clip(base_pan * pan_spread, -0.5, 0.5)
        pan      = base_pan + rng.uniform(-0.05, 0.05)
        l_gain   = np.sqrt(0.5 - pan * 0.5)
        r_gain   = np.sqrt(0.5 + pan * 0.5)

        end_samp = min(start_samp + len(tone), total_samples)
        length   = end_samp - start_samp
        stereo[start_samp:end_samp, 0] += tone[:length] * l_gain
        stereo[start_samp:end_samp, 1] += tone[:length] * r_gain

    return stereo


def write_wav(path: str, audio: np.ndarray):
    """Write float64 stereo array to a 16-bit stereo WAV file."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85          # headroom
    audio_int = (audio * 32767).astype(np.int16)

    with wave.open(path, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        # Interleave L/R
        interleaved = audio_int.flatten(order='C')
        wf.writeframes(interleaved.tobytes())
    print(f"Written: {path}  ({len(audio)/SAMPLE_RATE:.1f}s)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os

    print("Rendering melody …")
    all_events = MELODY + BASS
    total_beats = 56.0

    stereo = render_track(all_events, total_beats, reverb_room=0.55)

    out_path = os.path.join(os.path.dirname(__file__), OUTPUT_FILE)
    write_wav(out_path, stereo)
    print("Done.")
