"""
engine/synth.py
---------------
Bell tone synthesis engine.
- Additive synthesis with inharmonic partials
- Schroeder reverb via scipy (fast IIR)
- Key transposition and BPM support
"""

import io
import wave
import numpy as np
from scipy import signal as sp_signal

SAMPLE_RATE = 48_000

# Inharmonic bell partials: (freq_ratio, amplitude, decay_time_seconds)
BELL_TEXTURES: dict = {
    # Current default — bright, metallic tubular bell
    "tubular": [
        (1.000, 1.00, 2.8), (2.756, 0.55, 1.4),
        (5.404, 0.25, 0.7), (8.933, 0.12, 0.35), (13.34, 0.06, 0.18),
    ],
    # Deep, resonant church bell — more harmonic, long fundamental
    "church": [
        (1.000, 1.00, 4.5), (2.000, 0.60, 2.2),
        (3.000, 0.30, 1.1), (4.200, 0.15, 0.55), (5.400, 0.07, 0.28),
    ],
    # Tibetan singing bowl — very long decay, quasi-harmonic
    "bowl": [
        (1.000, 1.00, 6.5), (2.730, 0.45, 4.0),
        (4.900, 0.20, 2.5), (8.000, 0.10, 1.5),  (12.50, 0.04, 0.8),
    ],
    # Crystal / glass — high partials, glassy shimmer
    "crystal": [
        (1.000, 1.00, 3.5), (3.200, 0.50, 2.0),
        (6.800, 0.22, 1.0), (11.00, 0.10, 0.5),  (17.50, 0.04, 0.25),
    ],
}
_BASE_PARTIALS = BELL_TEXTURES["tubular"]  # kept for compatibility

_NOTE_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
_NOTE_ALIASES = {
    'C#': 'Db', 'D#': 'Eb', 'E#': 'F',
    'Gb': 'F#', 'G#': 'Ab', 'A#': 'Bb', 'B#': 'C',
}
_KEY_SEMITONES = {
    'C': 0, 'Db': 1, 'D': 2, 'Eb': 3, 'E': 4, 'F': 5,
    'F#': 6, 'G': 7, 'Ab': 8, 'A': 9, 'Bb': 10, 'B': 11,
    'C#': 1, 'D#': 3, 'G#': 8, 'A#': 10,
}


def note_freq(name: str) -> float:
    octave   = int(name[-1])
    note     = _NOTE_ALIASES.get(name[:-1], name[:-1])
    semitone = _NOTE_NAMES.index(note)
    midi     = (octave + 1) * 12 + semitone
    return 440.0 * 2 ** ((midi - 69) / 12)


def transpose_note(name: str, semitones: int) -> str:
    octave   = int(name[-1])
    note     = _NOTE_ALIASES.get(name[:-1], name[:-1])
    midi     = (octave + 1) * 12 + _NOTE_NAMES.index(note) + semitones
    new_oct  = midi // 12 - 1
    new_semi = midi % 12
    return _NOTE_NAMES[new_semi] + str(new_oct)


def transpose_events(events: list, semitones: int) -> list:
    if semitones == 0:
        return events
    return [(t, transpose_note(n, semitones), d, v) for t, n, d, v in events]


# ── Bell tone ─────────────────────────────────────────────────────────────────

def bell_tone(freq: float, duration: float, velocity: float = 1.0,
              decay_mult: float = 1.0, texture: str = "tubular",
              attack_ms: float = 0.0, beating: float = 0.0,
              strike_level: float = 0.0) -> np.ndarray:
    partials = BELL_TEXTURES.get(texture, BELL_TEXTURES["tubular"])
    rng = np.random.default_rng(int(freq * 137) % (2**31))
    n   = int(SAMPLE_RATE * duration)
    t   = np.linspace(0, duration, n, endpoint=False)
    sig = np.zeros(n, dtype=np.float64)
    for ratio, amp, tau in partials:
        detuning = rng.uniform(-0.003, 0.003) * beating
        env  = np.exp(-t / (tau * decay_mult))
        sig += amp * np.sin(2 * np.pi * freq * ratio * (1.0 + detuning) * t) * env
    # Soft-attack envelope
    if attack_ms > 0.0:
        ramp_n = min(int(attack_ms * SAMPLE_RATE / 1000), n)
        sig[:ramp_n] *= np.linspace(0.0, 1.0, ramp_n)
    # Strike transient
    if strike_level > 0.0:
        sig += _strike_transient(freq, n, strike_level, rng)
    peak = np.max(np.abs(sig)) or 1.0
    return sig / peak * velocity


def _strike_transient(freq: float, n: int, level: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Short bandpass noise burst simulating the mallet/clapper impact."""
    noise = rng.standard_normal(n)
    nyq   = SAMPLE_RATE / 2.0
    lo    = max(freq * 1.5, 300.0) / nyq
    hi    = min(freq * 6.0, 14000.0) / nyq
    if 0 < lo < hi < 0.99:
        try:
            sos   = sp_signal.butter(2, [lo, hi], 'bandpass', output='sos')
            noise = sp_signal.sosfilt(sos, noise)
        except Exception:
            pass
    t   = np.arange(n) / SAMPLE_RATE
    env = np.exp(-t / 0.004)                      # 4 ms decay
    peak = np.max(np.abs(noise)) or 1.0
    return noise / peak * env * level * 0.25


# ── Reverb (Schroeder via scipy IIR — fast) ───────────────────────────────────

def _comb(x: np.ndarray, delay_ms: float, gain: float) -> np.ndarray:
    d = int(delay_ms * SAMPLE_RATE / 1000)
    b = np.zeros(d + 1); b[0]  = 1.0
    a = np.zeros(d + 1); a[0]  = 1.0; a[-1] = -gain
    return sp_signal.lfilter(b, a, x)


def _allpass(x: np.ndarray, delay_ms: float, gain: float = 0.5) -> np.ndarray:
    d = int(delay_ms * SAMPLE_RATE / 1000)
    b = np.zeros(d + 1); b[0]  = -gain; b[-1] = 1.0
    a = np.zeros(d + 1); a[0]  =  1.0;  a[-1] = gain
    return sp_signal.lfilter(b, a, x)


def schroeder_reverb(signal: np.ndarray, room_scale: float = 0.5,
                     wet: float = 0.35) -> np.ndarray:
    g = 0.76 + 0.14 * room_scale
    combs = [
        _comb(signal, 29.7, g),
        _comb(signal, 37.1, g * 0.98),
        _comb(signal, 41.1, g * 0.96),
        _comb(signal, 43.7, g * 0.94),
    ]
    w = sum(combs) / len(combs)
    w = _allpass(w, 5.0)
    w = _allpass(w, 1.7)
    return signal * (1 - wet) + w * wet


# ── Freeverb (8 parallel damped combs + 4 series allpass) ────────────────────
# Delay lengths tuned for 48 kHz (scaled from the canonical 44.1 kHz values).
_FV_COMB_MS   = [25.31, 26.94, 28.96, 30.75, 32.25, 33.84, 35.28, 36.68]
_FV_ALLPASS_MS = [12.61, 10.00,  7.73,  5.10]
_FV_SPREAD_MS  = 0.52   # right-channel stereo spread


def _fast_comb(x: np.ndarray, delay_ms: float, gain: float) -> np.ndarray:
    """Undamped comb filter via d interleaved AR(1) processes — O(N) not O(N·d)."""
    d   = max(1, int(delay_ms * SAMPLE_RATE / 1000))
    n   = len(x)
    pad = (-n) % d
    xp  = np.pad(x, (0, pad))
    X   = xp.reshape(-1, d).T          # shape (d, n_blocks)
    b   = np.array([1.0])
    a   = np.array([1.0, -gain])
    for i in range(d):
        X[i] = sp_signal.lfilter(b, a, X[i])
    return X.T.reshape(-1)[:n]


def _fast_allpass(x: np.ndarray, delay_ms: float, gain: float = 0.5) -> np.ndarray:
    """Allpass filter via d interleaved AR(1) processes — O(N)."""
    d   = max(1, int(delay_ms * SAMPLE_RATE / 1000))
    n   = len(x)
    pad = (-n) % d
    xp  = np.pad(x, (0, pad))
    X   = xp.reshape(-1, d).T
    b   = np.array([-gain, 1.0])
    a   = np.array([1.0, -gain])
    for i in range(d):
        X[i] = sp_signal.lfilter(b, a, X[i])
    return X.T.reshape(-1)[:n]


def freeverb(stereo: np.ndarray, room_size: float = 0.5, damping: float = 0.4,
             wet: float = 0.35, width: float = 0.8) -> np.ndarray:
    """Freeverb: 8 damped-comb + 4 allpass, applied to stereo mix.
    room_size 0–1 → comb gain (decay time).
    damping   0–1 → high-freq absorption (post-comb 1-pole LP).
    wet       0–1 → dry/wet blend.
    width     0–1 → stereo spread."""
    gain = 0.70 + room_size * 0.28          # room_size→comb feedback gain
    mono = (stereo[:, 0] + stereo[:, 1]) * 0.5

    def _channel(spread: float) -> np.ndarray:
        combs = [_fast_comb(mono, d + spread, gain) for d in _FV_COMB_MS]
        rev   = sum(combs) * (1.0 / len(combs))
        # Post-comb damping (one-pole LP approximates in-loop damping)
        if damping > 0.01:
            lp_b = np.array([1.0 - damping])
            lp_a = np.array([1.0, -damping])
            rev  = sp_signal.lfilter(lp_b, lp_a, rev)
        for ap in _FV_ALLPASS_MS:
            rev = _fast_allpass(rev, ap + spread * 0.25)
        return rev

    rev_l = _channel(0.0)
    rev_r = _channel(_FV_SPREAD_MS)

    w2    = width / 2.0
    out_l = rev_l * (0.5 + w2) + rev_r * (0.5 - w2)
    out_r = rev_r * (0.5 + w2) + rev_l * (0.5 - w2)

    dry = 1.0 - wet
    return np.column_stack([stereo[:, 0] * dry + out_l * wet,
                            stereo[:, 1] * dry + out_r * wet])


# ── Stereo delay ──────────────────────────────────────────────────────────────

def apply_delay(stereo: np.ndarray, delay_ms: float = 300.0,
                feedback: float = 0.35, wet: float = 0.0) -> np.ndarray:
    """Feedback echo delay. Adds layered echoes spaced delay_ms apart."""
    if wet < 0.005 or delay_ms < 1.0 or feedback < 0.005:
        return stereo
    d = int(delay_ms * SAMPLE_RATE / 1000)
    if d >= len(stereo):
        return stereo
    n   = len(stereo)
    out = stereo.astype(np.float64, copy=True)
    # Right channel offset for stereo ping-pong feel
    d_r = max(1, int(delay_ms * 0.97 * SAMPLE_RATE / 1000))
    delays = [d, d_r]
    fb = feedback * wet
    layer = stereo.astype(np.float64)
    for echo_n in range(16):                     # up to 16 echoes
        if fb < 0.0005:
            break
        new_layer = np.zeros_like(layer)
        dl = delays[echo_n % 2]
        new_layer[dl:] = layer[:n - dl]
        out += new_layer * fb
        layer = new_layer
        fb   *= feedback
    # Hard-limit to prevent runaway feedback
    peak = np.max(np.abs(out))
    if peak > 1.0:
        out /= peak
    return out




def apply_octave_spread(events: list, spread: float, seed: int = 42) -> list:
    """Randomly shift notes up/down by octaves. spread 0=off, 3=wild."""
    if spread <= 0:
        return list(events)
    rng = np.random.default_rng(abs(seed))
    max_oct  = min(max(1, int(np.ceil(spread))), 3)
    prob     = min(0.9, spread / 3.0)
    result   = []
    for beat, note, dur, vel in events:
        if rng.random() < prob:
            shift = int(rng.integers(1, max_oct + 1)) * 12
            if rng.random() > 0.5:
                shift = -shift
            try:
                new_note = transpose_note(note, shift)
                while int(new_note[-1]) < 2:
                    new_note = transpose_note(new_note, 12)
                while int(new_note[-1]) > 6:
                    new_note = transpose_note(new_note, -12)
                note = new_note
            except Exception:
                pass
        result.append((beat, note, dur, vel))
    return result


# ── Track renderer ─────────────────────────────────────────────────────────────

def render_track(events: list, total_beats: float,
                 bpm: float = 60.0, reverb_room: float = 0.5,
                 reverb_damping: float = 0.4, reverb_width: float = 0.8,
                 reverb_wet: float = 0.35,
                 decay_mult: float = 1.0, pan_spread: float = 0.4,
                 key_semitones: int = 0, octave_spread: float = 0.0,
                 base_octave_shift: int = 0, humanize: float = 0.0,
                 density: float = 1.0, texture: str = "tubular",
                 attack_ms: float = 0.0, beating: float = 0.0,
                 strike_level: float = 0.0,
                 delay_time: float = 300.0, delay_feedback: float = 0.35,
                 delay_wet: float = 0.0, time_scatter: float = 0.0) -> np.ndarray:
    """Render events to a stereo float64 array with global Freeverb + Delay."""
    beat_dur     = 60.0 / bpm
    total_samp   = int(total_beats * beat_dur * SAMPLE_RATE) + SAMPLE_RATE * 6
    stereo       = np.zeros((total_samp, 2), dtype=np.float64)
    rng          = np.random.default_rng(42)

    events = transpose_events(events, key_semitones + base_octave_shift * 12)
    events = apply_octave_spread(events, octave_spread, seed=42)

    for beat, note, dur_beats, vel in events:
        if density < 1.0 and rng.random() > density:
            continue
        freq       = note_freq(note)
        dur_s      = max(dur_beats * beat_dur, 0.05) + 3.0 * decay_mult
        jitter     = rng.uniform(-0.025, 0.025) * humanize
        grid_t     = beat * beat_dur + jitter
        if time_scatter > 0:
            rand_t = rng.uniform(0, total_beats * beat_dur)
            t      = grid_t * (1 - time_scatter) + rand_t * time_scatter
        else:
            t = grid_t
        start      = max(0, int(t * SAMPLE_RATE))
        vel_scaled = float(np.clip(vel * rng.uniform(1 - 0.25 * humanize, 1 + 0.1 * humanize), 0.05, 1.0))
        note_atk   = attack_ms * rng.uniform(0.5, 1.5) if humanize > 0 else attack_ms
        tone       = bell_tone(freq, dur_s, velocity=vel_scaled, decay_mult=decay_mult,
                               texture=texture, attack_ms=note_atk,
                               beating=beating, strike_level=strike_level)

        base_pan   = np.clip((freq - 400) / 1200 * pan_spread, -0.5, 0.5)
        pan        = base_pan + rng.uniform(-0.05, 0.05)
        l_g        = np.sqrt(0.5 - pan * 0.5)
        r_g        = np.sqrt(0.5 + pan * 0.5)

        end        = min(start + len(tone), total_samp)
        length     = end - start
        stereo[start:end, 0] += tone[:length] * l_g
        stereo[start:end, 1] += tone[:length] * r_g

    out = freeverb(stereo, room_size=reverb_room, damping=reverb_damping,
                   wet=reverb_wet, width=reverb_width)
    return apply_delay(out, delay_ms=delay_time, feedback=delay_feedback, wet=delay_wet)


_LOOKAHEAD_BEATS = 8.0  # look back this many beats for reverb tails (gapless looping)


def render_chunk(events: list, beat_offset: float, chunk_beats: float,
                 total_beats: float, bpm: float = 60.0,
                 reverb_room: float = 0.5, reverb_damping: float = 0.4,
                 reverb_width: float = 0.8, reverb_wet: float = 0.35,
                 decay_mult: float = 1.0, pan_spread: float = 0.4,
                 key_semitones: int = 0, octave_spread: float = 0,
                 base_octave_shift: int = 0, humanize: float = 0.0,
                 density: float = 1.0, texture: str = "tubular",
                 attack_ms: float = 0.0, beating: float = 0.0,
                 strike_level: float = 0.0, seed: int = 42,
                 delay_time: float = 300.0, delay_feedback: float = 0.35,
                 delay_wet: float = 0.0, time_scatter: float = 0.0) -> np.ndarray:
    """Render chunk_beats starting at beat_offset with gapless lookahead, Freeverb + Delay."""
    beat_dur    = 60.0 / bpm
    chunk_end   = beat_offset + chunk_beats

    window: list = []
    for b, n, d, v in events:
        loop_b     = b % total_beats
        iterations = int(beat_offset // total_beats)
        for iter_off in (0, -1, 1):
            abs_b = loop_b + (iterations + iter_off) * total_beats
            rel   = abs_b - beat_offset
            if -_LOOKAHEAD_BEATS <= rel < chunk_beats:
                window.append((rel, n, d, v))
                break

    lookahead_evts = [(r, n, d, v) for r, n, d, v in window if r < 0]
    normal_evts    = [(r, n, d, v) for r, n, d, v in window if r >= 0]

    normal_evts    = transpose_events(normal_evts, key_semitones + base_octave_shift * 12)
    loop_seed      = seed + int(beat_offset // total_beats) * 999
    normal_evts    = apply_octave_spread(normal_evts, octave_spread, seed=loop_seed)

    lookahead_evts = transpose_events(lookahead_evts, key_semitones + base_octave_shift * 12)
    prev_seed      = seed + max(0, int((beat_offset - 1) // total_beats)) * 999
    lookahead_evts = apply_octave_spread(lookahead_evts, octave_spread, seed=prev_seed)

    reverb_tail = 4.5 * max(decay_mult, 1.0)
    total_samp  = int((chunk_beats * beat_dur + reverb_tail) * SAMPLE_RATE) + SAMPLE_RATE
    stereo      = np.zeros((total_samp, 2), dtype=np.float64)
    rng         = np.random.default_rng(seed + 300)
    # Per-chunk scatter seed — reproducible per chunk but different across chunks
    scatter_rng = np.random.default_rng(abs(seed + int(beat_offset * 100)))

    for rel_beat, note, dur_beats, vel in lookahead_evts + normal_evts:
        is_normal = rel_beat >= 0
        if is_normal and density < 1.0 and rng.random() > density:
            continue
        freq       = note_freq(note)
        dur_s      = max(dur_beats * beat_dur, 0.05) + 3.0 * decay_mult
        jitter     = rng.uniform(-0.025, 0.025) * humanize if is_normal else 0.0
        vel_scaled = float(np.clip(
            vel * (rng.uniform(1 - 0.25 * humanize, 1 + 0.1 * humanize) if is_normal else 1.0),
            0.05, 1.0))
        note_atk   = attack_ms * rng.uniform(0.5, 1.5) if (humanize > 0 and is_normal) else attack_ms
        tone       = bell_tone(freq, dur_s, velocity=vel_scaled, decay_mult=decay_mult,
                               texture=texture, attack_ms=note_atk,
                               beating=beating, strike_level=strike_level if is_normal else 0.0)

        if is_normal and time_scatter > 0:
            # Interpolate between grid position and a random position within the chunk
            grid_t  = rel_beat * beat_dur + jitter
            rand_t  = scatter_rng.uniform(0.0, chunk_beats * beat_dur)
            t       = grid_t * (1.0 - time_scatter) + rand_t * time_scatter
            start_samp = max(0, int(t * SAMPLE_RATE))
        else:
            start_samp = int((rel_beat * beat_dur + jitter) * SAMPLE_RATE)

        if start_samp < 0:
            skip = -start_samp
            if skip >= len(tone):
                continue
            tone       = tone[skip:]
            start_samp = 0

        base_pan = np.clip((freq - 400) / 1200 * pan_spread, -0.5, 0.5)
        pan      = float(base_pan + rng.uniform(-0.05, 0.05))
        l_g      = np.sqrt(0.5 - pan * 0.5)
        r_g      = np.sqrt(0.5 + pan * 0.5)

        end    = min(start_samp + len(tone), total_samp)
        length = end - start_samp
        if length > 0:
            stereo[start_samp:end, 0] += tone[:length] * l_g
            stereo[start_samp:end, 1] += tone[:length] * r_g

    out = freeverb(stereo, room_size=reverb_room, damping=reverb_damping,
                   wet=reverb_wet, width=reverb_width)
    return apply_delay(out, delay_ms=delay_time, feedback=delay_feedback, wet=delay_wet)
    """Render chunk_beats starting at beat_offset with gapless lookahead and global Freeverb."""
    beat_dur    = 60.0 / bpm
    chunk_end   = beat_offset + chunk_beats

    window: list = []
    for b, n, d, v in events:
        loop_b     = b % total_beats
        iterations = int(beat_offset // total_beats)
        for iter_off in (0, -1, 1):
            abs_b = loop_b + (iterations + iter_off) * total_beats
            rel   = abs_b - beat_offset
            if -_LOOKAHEAD_BEATS <= rel < chunk_beats:
                window.append((rel, n, d, v))
                break

    lookahead_evts = [(r, n, d, v) for r, n, d, v in window if r < 0]
    normal_evts    = [(r, n, d, v) for r, n, d, v in window if r >= 0]

    normal_evts    = transpose_events(normal_evts, key_semitones + base_octave_shift * 12)
    loop_seed      = seed + int(beat_offset // total_beats) * 999
    normal_evts    = apply_octave_spread(normal_evts, octave_spread, seed=loop_seed)

    lookahead_evts = transpose_events(lookahead_evts, key_semitones + base_octave_shift * 12)
    prev_seed      = seed + max(0, int((beat_offset - 1) // total_beats)) * 999
    lookahead_evts = apply_octave_spread(lookahead_evts, octave_spread, seed=prev_seed)

    reverb_tail = 4.5 * max(decay_mult, 1.0)
    total_samp  = int((chunk_beats * beat_dur + reverb_tail) * SAMPLE_RATE) + SAMPLE_RATE
    stereo      = np.zeros((total_samp, 2), dtype=np.float64)
    rng         = np.random.default_rng(seed + 300)

    for rel_beat, note, dur_beats, vel in lookahead_evts + normal_evts:
        is_normal = rel_beat >= 0
        if is_normal and density < 1.0 and rng.random() > density:
            continue
        freq       = note_freq(note)
        dur_s      = max(dur_beats * beat_dur, 0.05) + 3.0 * decay_mult
        jitter     = rng.uniform(-0.025, 0.025) * humanize if is_normal else 0.0
        vel_scaled = float(np.clip(
            vel * (rng.uniform(1 - 0.25 * humanize, 1 + 0.1 * humanize) if is_normal else 1.0),
            0.05, 1.0))
        note_atk   = attack_ms * rng.uniform(0.5, 1.5) if (humanize > 0 and is_normal) else attack_ms
        tone       = bell_tone(freq, dur_s, velocity=vel_scaled, decay_mult=decay_mult,
                               texture=texture, attack_ms=note_atk,
                               beating=beating, strike_level=strike_level if is_normal else 0.0)

        start_samp = int((rel_beat * beat_dur + jitter) * SAMPLE_RATE)
        if start_samp < 0:
            skip = -start_samp
            if skip >= len(tone):
                continue
            tone       = tone[skip:]
            start_samp = 0

        base_pan = np.clip((freq - 400) / 1200 * pan_spread, -0.5, 0.5)
        pan      = float(base_pan + rng.uniform(-0.05, 0.05))
        l_g      = np.sqrt(0.5 - pan * 0.5)
        r_g      = np.sqrt(0.5 + pan * 0.5)

        end    = min(start_samp + len(tone), total_samp)
        length = end - start_samp
        if length > 0:
            stereo[start_samp:end, 0] += tone[:length] * l_g
            stereo[start_samp:end, 1] += tone[:length] * r_g

    return freeverb(stereo, room_size=reverb_room, damping=reverb_damping,
                    wet=reverb_wet, width=reverb_width)


def stereo_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Normalise and encode stereo float64 array to 16-bit WAV bytes."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.85
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.flatten(order='C').tobytes())
    return buf.getvalue()
