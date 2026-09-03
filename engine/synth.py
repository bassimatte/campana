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
    # Large cast-bronze bell. The sub-fundamental hum and clustered tierce,
    # quint, and nominal partials give Cattedrale weight without relying on EQ.
    "bronze": [
        (0.500, 0.42, 8.0), (1.000, 1.00, 6.5), (1.198, 0.62, 4.8),
        (1.506, 0.42, 3.8), (2.000, 0.70, 3.2), (2.416, 0.30, 2.4),
        (3.010, 0.22, 1.7), (4.080, 0.15, 1.0), (5.430, 0.09, 0.55),
        (7.120, 0.05, 0.28),
    ],
    # Small handbell/carillon voice: fast upper-partial decay and a clear,
    # bright nominal make dense Festa patterns articulate rather than smear.
    "handbell": [
        (1.000, 1.00, 2.2), (1.190, 0.38, 1.45), (1.505, 0.24, 1.0),
        (2.000, 0.52, 0.85), (2.720, 0.26, 0.55), (3.980, 0.18, 0.34),
        (5.620, 0.10, 0.20), (8.100, 0.05, 0.11),
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


# ── Scale quantization ────────────────────────────────────────────────────────

SCALES: dict = {
    "minor":      [0, 2, 3, 5, 7, 8, 10],   # natural minor (default)
    "major":      [0, 2, 4, 5, 7, 9, 11],
    "dorian":     [0, 2, 3, 5, 7, 9, 10],
    "pentatonic": [0, 3, 5, 7, 10],          # minor pentatonic
    "whole_tone": [0, 2, 4, 6, 8, 10],
}


def _snap_midi_to_scale(midi: int, root_class: int, degrees: list) -> int:
    """Return the MIDI note number closest to midi that lies in the scale."""
    offset = (midi - root_class) % 12
    best, best_dist = degrees[0], 12
    for d in degrees:
        dist = min(abs(d - offset), 12 - abs(d - offset))
        if dist < best_dist:
            best, best_dist = d, dist
    delta = best - offset
    if delta > 6:
        delta -= 12
    if delta < -6:
        delta += 12
    return midi + delta


def apply_scale_quantize(events: list, root_semitone: int, scale: str) -> list:
    """Snap all note pitches to the nearest degree of the given scale.
    No-op when scale is 'minor' (presets are written in natural minor)."""
    degrees = SCALES.get(scale)
    if not degrees or scale == "minor":
        return events
    root_class = root_semitone % 12
    result = []
    for beat, note, dur, vel in events:
        try:
            octave  = int(note[-1])
            n_name  = _NOTE_ALIASES.get(note[:-1], note[:-1])
            midi    = (octave + 1) * 12 + _NOTE_NAMES.index(n_name)
            new_mid = _snap_midi_to_scale(midi, root_class, degrees)
            note    = _NOTE_NAMES[new_mid % 12] + str(new_mid // 12 - 1)
        except Exception:
            pass
        result.append((beat, note, dur, vel))
    return result


# ── Bell tone ─────────────────────────────────────────────────────────────────

def bell_tone(freq: float, duration: float, velocity: float = 1.0,
              decay_mult: float = 1.0, texture: str = "tubular",
              attack_ms: float = 0.0, beating: float = 0.0,
              strike_level: float = 0.0,
              variation_seed=None) -> np.ndarray:
    partials = BELL_TEXTURES.get(texture, BELL_TEXTURES["tubular"])
    n_partials = len(partials)
    seed = int(freq * 137) if variation_seed is None else int(variation_seed + freq * 137)
    rng = np.random.default_rng(abs(seed) % (2**31))
    n   = int(SAMPLE_RATE * duration)
    t   = np.linspace(0, duration, n, endpoint=False, dtype=np.float32)
    sig = np.zeros(n, dtype=np.float32)

    # Perceptual velocity curve — soft notes feel softer, hard notes have more punch
    vel_eff = float(np.clip(velocity ** 0.65, 0.05, 1.0))

    for i, (ratio, amp, tau) in enumerate(partials):
        # Velocity-dependent brightness: harder strikes excite higher partials more.
        brightness = 1.0 + (vel_eff - 0.5) * 2.0 * (i / max(1, n_partials - 1)) * 0.7
        eff_amp    = amp * max(0.02, brightness)
        detuning   = rng.uniform(-0.0015, 0.0015) * beating
        env  = np.exp(-t / (tau * decay_mult))
        partial_freq = freq * ratio * (1.0 + detuning)
        if partial_freq >= SAMPLE_RATE * 0.48:
            continue
        if beating > 0.0:
            # A close oscillator pair creates actual slow amplitude beating.
            spread = beating * (0.00012 + i * 0.00008)
            wave = 0.5 * (
                np.sin(2 * np.pi * partial_freq * (1.0 - spread) * t) +
                np.sin(2 * np.pi * partial_freq * (1.0 + spread) * t)
            )
        else:
            wave = np.sin(2 * np.pi * partial_freq * t)
        sig += eff_amp * wave * env
    # Soft-attack envelope
    if attack_ms > 0.0:
        ramp_n = min(int(attack_ms * SAMPLE_RATE / 1000), n)
        sig[:ramp_n] *= np.linspace(0.0, 1.0, ramp_n)
    # Strike transient: auto-scales with velocity so hard notes feel physical
    effective_strike = strike_level + vel_eff * 0.12
    if effective_strike > 0.01:
        sig += _strike_transient(freq, n, effective_strike, rng)
    peak = np.max(np.abs(sig)) or 1.0
    sig *= vel_eff / peak
    return sig


def _strike_transient(freq: float, n: int, level: float,
                      rng: np.random.Generator) -> np.ndarray:
    """Short bandpass noise burst simulating the mallet/clapper impact."""
    noise = rng.standard_normal(n, dtype=np.float32)
    nyq   = SAMPLE_RATE / 2.0
    lo    = max(freq * 1.5, 300.0) / nyq
    hi    = min(freq * 6.0, 14000.0) / nyq
    if 0 < lo < hi < 0.99:
        try:
            sos   = sp_signal.butter(2, [lo, hi], 'bandpass', output='sos')
            noise = sp_signal.sosfilt(sos.astype(np.float32), noise)
        except Exception:
            pass
    t   = np.arange(n, dtype=np.float32) / SAMPLE_RATE
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
    b   = np.array([1.0], dtype=x.dtype)
    a   = np.array([1.0, -gain], dtype=x.dtype)
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
    b   = np.array([-gain, 1.0], dtype=x.dtype)
    a   = np.array([1.0, -gain], dtype=x.dtype)
    for i in range(d):
        X[i] = sp_signal.lfilter(b, a, X[i])
    return X.T.reshape(-1)[:n]


def freeverb(stereo: np.ndarray, room_size: float = 0.5, damping: float = 0.4,
             wet: float = 0.35, width: float = 0.8, shimmer: float = 0.0) -> np.ndarray:
    """Freeverb: 8 damped-comb + 4 allpass, applied to stereo mix.
    room_size 0–1 → comb gain (decay time).
    damping   0–1 → high-freq absorption (post-comb 1-pole LP).
    wet       0–1 → dry/wet blend.
    width     0–1 → stereo spread.
    shimmer   0–1 → blend in octave-up pitch-shifted reverb tail."""
    gain = 0.70 + room_size * 0.28          # room_size→comb feedback gain
    mono = (stereo[:, 0] + stereo[:, 1]) * 0.5

    def _channel(spread: float) -> np.ndarray:
        # Accumulate combs one at a time to avoid holding 8×45 MB simultaneously
        rev = np.zeros(len(mono), dtype=mono.dtype)
        for d in _FV_COMB_MS:
            rev += _fast_comb(mono, d + spread, gain)
        rev *= (1.0 / len(_FV_COMB_MS))
        # Post-comb damping (one-pole LP approximates in-loop damping)
        if damping > 0.01:
            lp_b = np.array([1.0 - damping], dtype=rev.dtype)
            lp_a = np.array([1.0, -damping], dtype=rev.dtype)
            rev  = sp_signal.lfilter(lp_b, lp_a, rev)
        for ap in _FV_ALLPASS_MS:
            rev = _fast_allpass(rev, ap + spread * 0.25)
        return rev

    rev_l = _channel(0.0)
    rev_r = _channel(_FV_SPREAD_MS)
    del mono   # free before building output

    # Shimmer: add octave-up pitch-shifted version of the reverb wet signal.
    # resample_poly(x, 1, 2) → half the sample count = one octave higher pitch.
    # The shorter buffer is placed at t=0; it decays twice as fast as the base
    # reverb, which is physically realistic (higher partials decay sooner).
    if shimmer > 0.005:
        n    = len(rev_l)
        sh_l = sp_signal.resample_poly(rev_l, 1, 2)
        sh_r = sp_signal.resample_poly(rev_r, 1, 2)
        n_sh = min(len(sh_l), n)
        buf_l = np.zeros(n, dtype=rev_l.dtype); buf_l[:n_sh] = sh_l[:n_sh]
        buf_r = np.zeros(n, dtype=rev_r.dtype); buf_r[:n_sh] = sh_r[:n_sh]
        del sh_l, sh_r
        rev_l = rev_l + buf_l * (shimmer * 0.45)
        rev_r = rev_r + buf_r * (shimmer * 0.45)
        del buf_l, buf_r

    # Build output in a pre-allocated array using in-place ops to minimise
    # peak memory — avoids materialising out_l and out_r as separate arrays
    # while column_stack output and stereo are all simultaneously live.
    n     = len(rev_l)
    dry   = 1.0 - wet
    w2    = width / 2.0
    out   = np.empty((n, 2), dtype=rev_l.dtype)

    # out[:,0] = (rev_l*(0.5+w2) + rev_r*(0.5-w2))*wet + stereo[:,0]*dry
    np.multiply(rev_l, 0.5 + w2, out=out[:, 0])
    out[:, 0] += rev_r * (0.5 - w2)
    out[:, 0] *= wet
    out[:, 0] += stereo[:, 0] * dry

    # out[:,1] = (rev_r*(0.5+w2) + rev_l*(0.5-w2))*wet + stereo[:,1]*dry
    np.multiply(rev_r, 0.5 + w2, out=out[:, 1])
    out[:, 1] += rev_l * (0.5 - w2)
    del rev_l, rev_r   # free 2×45 MB before the last addition
    out[:, 1] *= wet
    out[:, 1] += stereo[:, 1] * dry

    return out


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
    out = stereo.copy()
    # Right channel offset for stereo ping-pong feel
    d_r = max(1, int(delay_ms * 0.97 * SAMPLE_RATE / 1000))
    delays = [d, d_r]
    fb = feedback * wet
    layer = stereo.copy()
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
                 delay_wet: float = 0.0, time_scatter: float = 0.0,
                 shimmer: float = 0.0, scale_mode: str = "minor") -> np.ndarray:
    """Render events to a stereo float64 array with global Freeverb + Delay."""
    beat_dur     = 60.0 / bpm
    total_samp   = int(total_beats * beat_dur * SAMPLE_RATE) + SAMPLE_RATE * 6
    stereo       = np.zeros((total_samp, 2), dtype=np.float32)
    rng          = np.random.default_rng(42)

    events = transpose_events(events, key_semitones + base_octave_shift * 12)
    events = apply_scale_quantize(events, key_semitones, scale_mode)
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
        strike_seed = _strike_variation_seed(beat)
        tone       = bell_tone(freq, dur_s, velocity=vel_scaled, decay_mult=decay_mult,
                               texture=texture, attack_ms=note_atk,
                               beating=beating, strike_level=strike_level,
                               variation_seed=strike_seed)

        base_pan   = np.clip((freq - 400) / 1200 * pan_spread, -0.5, 0.5)
        pan        = base_pan + rng.uniform(-0.05, 0.05)
        l_g        = np.sqrt(0.5 - pan * 0.5)
        r_g        = np.sqrt(0.5 + pan * 0.5)

        end        = min(start + len(tone), total_samp)
        length     = end - start
        stereo[start:end, 0] += tone[:length] * l_g
        stereo[start:end, 1] += tone[:length] * r_g

    out = freeverb(stereo, room_size=reverb_room, damping=reverb_damping,
                   wet=reverb_wet, width=reverb_width, shimmer=shimmer)
    return apply_delay(out, delay_ms=delay_time, feedback=delay_feedback, wet=delay_wet)


# ── Generative melody engine ──────────────────────────────────────────────────

def generate_bell_events(scale_mode: str, total_beats: float,
                         gen_params: dict, seed: int) -> list:
    """Procedurally generate one loop-cycle of bell events.

    Notes are produced in C (key_semitones=0) so that render_chunk's
    existing transpose_events() call moves them into the user's chosen key.

    gen_params keys:
      style                 "walk" (default), "tolling", or "carillon"
      melody_octaves       list of MIDI octave numbers for the melody voice
      bass_octaves         list of MIDI octave numbers for the bass voice
      note_spacing_range   (lo, hi) beats between successive melody strikes
      bass_spacing_range   (lo, hi) beats between successive bass strikes
      velocity_base        base velocity for melody (0-1)
      bass_velocity_base   base velocity for bass (0-1)
      walk_bias            -1 descending / 0 neutral / 1 ascending
      bass_enabled         bool

    Specialized styles additionally consume toll_pattern and
    response_probability (tolling), or variation_probability (carillon).
    """
    rng     = np.random.default_rng(seed)
    degrees = SCALES.get(scale_mode, SCALES["minor"])

    def build_pool(octaves):
        pool = []
        for oct in octaves:
            for d in degrees:
                pool.append((oct + 1) * 12 + d)   # root = C
        return sorted(set(pool))

    def midi_to_name(midi: int) -> str:
        midi = int(np.clip(midi, 12, 107))
        return _NOTE_NAMES[midi % 12] + str(midi // 12 - 1)

    melody_pool = build_pool(gen_params.get("melody_octaves", [3, 4, 5]))
    bass_pool   = build_pool(gen_params.get("bass_octaves",   [1, 2]))

    events: list = []

    style = gen_params.get("style", "walk")

    if style == "tolling":
        # A sparse, low-register ritual grammar for Cattedrale. Each 16-beat
        # epoch keeps the same root/fifth/tierce contour but varies weight,
        # timing, and the occasional distant response.
        toll_pattern = gen_params.get("toll_pattern", [0.0, 6.5, 13.0])
        octaves = gen_params.get("melody_octaves", [2, 3])
        velocity_base = gen_params.get("velocity_base", 0.82)
        response_probability = gen_params.get("response_probability", 0.25)
        degree_positions = [0, 4, 2]  # root, fifth, tierce in a heptatonic scale

        for strike_index, nominal_beat in enumerate(toll_pattern):
            beat = max(0.0, float(nominal_beat + rng.uniform(-0.16, 0.16)))
            if beat >= total_beats:
                continue
            degree = degrees[degree_positions[strike_index % len(degree_positions)] % len(degrees)]
            octave = octaves[0] if strike_index != 1 else octaves[min(1, len(octaves) - 1)]
            note = midi_to_name((octave + 1) * 12 + degree)
            duration = float(rng.uniform(4.5, 7.5))
            velocity = float(np.clip(
                velocity_base + (0.08 if strike_index == 0 else 0.0) + rng.uniform(-0.05, 0.05),
                0.45, 1.0))
            events.append((beat, note, duration, velocity))

            if rng.random() < response_probability:
                response_beat = beat + float(rng.uniform(1.35, 2.15))
                if response_beat < total_beats:
                    response_degree = degrees[(degree_positions[strike_index % 3] + 2) % len(degrees)]
                    response_octave = octaves[min(1, len(octaves) - 1)]
                    response_note = midi_to_name((response_octave + 1) * 12 + response_degree)
                    events.append((response_beat, response_note, 2.5,
                                   float(np.clip(velocity * 0.58, 0.25, 0.65))))
        return sorted(events, key=lambda e: e[0])

    if style == "carillon":
        # A repeating four-bar harmonic sentence for Festa. The recognizable
        # arpeggio survives from epoch to epoch while small seeded variations
        # keep the carillon alive rather than looped.
        melody_octaves = gen_params.get("melody_octaves", [4, 5, 6])
        bass_octaves = gen_params.get("bass_octaves", [3, 4])
        velocity_base = gen_params.get("velocity_base", 0.72)
        bass_velocity = gen_params.get("bass_velocity_base", 0.52)
        variation_probability = gen_params.get("variation_probability", 0.35)
        progression = [0, 3, 4, 0]  # I – IV – V – I
        base_pattern = [0, 1, 2, 3, 2, 1, 2, 3]

        def scale_midi(degree_position: int, octave: int) -> int:
            octave_add, degree_index = divmod(degree_position, len(degrees))
            return (octave + octave_add + 1) * 12 + degrees[degree_index]

        bar = 0
        bar_start = 0.0
        while bar_start < total_beats:
            root_position = progression[bar % len(progression)]
            chord_positions = [
                root_position, root_position + 2, root_position + 4,
                root_position + len(degrees),
            ]
            pattern = list(base_pattern)
            if rng.random() < variation_probability:
                if rng.random() < 0.5:
                    pattern[4:] = reversed(pattern[4:])
                else:
                    pattern = pattern[2:] + pattern[:2]

            melody_octave = melody_octaves[min(bar % 2, len(melody_octaves) - 1)]
            for step_index, chord_index in enumerate(pattern):
                beat = bar_start + step_index * 0.5
                if beat >= total_beats:
                    break
                midi = scale_midi(chord_positions[chord_index], melody_octave)
                velocity = float(np.clip(
                    velocity_base + (0.14 if step_index in (0, 6) else 0.0) +
                    rng.uniform(-0.04, 0.04), 0.35, 1.0))
                events.append((beat, midi_to_name(midi), 0.42, velocity))

            if gen_params.get("bass_enabled", True):
                bass_octave = bass_octaves[bar % len(bass_octaves)]
                for beat_offset, degree_offset in ((0.0, 0), (2.0, 4)):
                    beat = bar_start + beat_offset
                    if beat < total_beats:
                        midi = scale_midi(root_position + degree_offset, bass_octave)
                        events.append((beat, midi_to_name(midi), 1.4,
                                       float(np.clip(bass_velocity + rng.uniform(-0.03, 0.03),
                                                     0.25, 0.75))))
            bar += 1
            bar_start += 4.0
        return sorted(events, key=lambda e: e[0])

    def walk_step(rng, pos: int, pool_size: int, steps, weights,
                  leap_prob: float = 0.12) -> int:
        """Random walk with boundary bouncing and occasional leaps.
        Never clips at edges (which causes clustering), instead bounces.
        Occasional random leap prevents long-term clustering."""
        if rng.random() < leap_prob:
            return int(rng.integers(0, pool_size))
        step    = int(rng.choice(steps, p=weights))
        new_pos = pos + step
        # Bounce off boundaries instead of clamping
        if new_pos < 0:
            new_pos = abs(new_pos)
        elif new_pos >= pool_size:
            new_pos = 2 * pool_size - new_pos - 2
        return int(np.clip(new_pos, 0, pool_size - 1))

    # ── Melody voice ──────────────────────────────────────────────────────────
    sp_lo, sp_hi = gen_params.get("note_spacing_range", (0.75, 2.0))
    vel_base     = gen_params.get("velocity_base", 0.65)
    bias         = gen_params.get("walk_bias", 0)

    # No step=0 so the note always changes; ±3 allows broader leaps
    mel_steps = np.array([-3, -2, -1, 1, 2, 3])
    if bias > 0:
        mel_w = np.array([1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
    elif bias < 0:
        mel_w = np.array([2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
    else:
        mel_w = np.array([2.0, 3.0, 4.0, 4.0, 3.0, 2.0])
    mel_w /= mel_w.sum()

    mel_size = len(melody_pool)
    pos  = int(rng.integers(0, mel_size))  # random start per epoch, not always middle
    beat = float(rng.uniform(0, max(0.01, sp_lo * 0.5)))
    while beat < total_beats:
        pos  = walk_step(rng, pos, mel_size, mel_steps, mel_w)
        note = midi_to_name(melody_pool[pos])
        vel  = float(np.clip(
            vel_base + 0.3 * (pos / max(1, mel_size - 1)) + rng.uniform(-0.08, 0.08),
            0.15, 1.0))
        dur  = float(rng.choice([0.5, 0.5, 1.0, 1.0, 2.0]))
        events.append((beat, note, dur, vel))
        beat += float(rng.uniform(sp_lo, sp_hi))

    # ── Bass voice ────────────────────────────────────────────────────────────
    if gen_params.get("bass_enabled", True) and bass_pool:
        sp_lo2, sp_hi2 = gen_params.get("bass_spacing_range", (2.0, 5.0))
        vel_b2         = gen_params.get("bass_velocity_base", 0.50)
        bas_steps = np.array([-2, -1, 1, 2])
        bas_w     = np.array([1.0, 2.0, 2.0, 1.0])
        bas_w    /= bas_w.sum()
        bas_size  = len(bass_pool)
        pos2  = int(rng.integers(0, bas_size))  # random start per epoch
        beat2 = float(rng.uniform(0, sp_lo2))
        while beat2 < total_beats:
            pos2  = walk_step(rng, pos2, bas_size, bas_steps, bas_w, leap_prob=0.08)
            note2 = midi_to_name(bass_pool[pos2])
            vel2  = float(np.clip(
                vel_b2 + 0.2 * (pos2 / max(1, bas_size - 1)) + rng.uniform(-0.05, 0.05),
                0.10, 0.80))
            dur2  = float(rng.choice([1.0, 2.0, 2.0, 4.0]))
            events.append((beat2, note2, dur2, vel2))
            beat2 += float(rng.uniform(sp_lo2, sp_hi2))

    return sorted(events, key=lambda e: e[0])


_LOOKAHEAD_BEATS = 8.0  # look back this many beats for reverb tails (gapless looping)
_STEREO_MICRO    = 0.0012  # L/R frequency offset ≈ 2 cents → ~0.3–2.5 Hz natural beating


def _strike_variation_seed(absolute_beat: float) -> int:
    """Stable timbre seed for a strike, including when its tail is regenerated."""
    return int(round(absolute_beat * 1000.0)) * 7919


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
                 delay_wet: float = 0.0, time_scatter: float = 0.0,
                 shimmer: float = 0.0, scale_mode: str = "minor") -> np.ndarray:
    """Render chunk_beats starting at beat_offset with gapless lookahead, Freeverb + Delay."""
    beat_dur    = 60.0 / bpm

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

    key_shift      = key_semitones + base_octave_shift * 12
    normal_evts    = transpose_events(normal_evts, key_shift)
    normal_evts    = apply_scale_quantize(normal_evts, key_semitones, scale_mode)
    loop_seed      = seed + int(beat_offset // total_beats) * 999
    normal_evts    = apply_octave_spread(normal_evts, octave_spread, seed=loop_seed)

    lookahead_evts = transpose_events(lookahead_evts, key_shift)
    lookahead_evts = apply_scale_quantize(lookahead_evts, key_semitones, scale_mode)
    prev_seed      = seed + max(0, int((beat_offset - 1) // total_beats)) * 999
    lookahead_evts = apply_octave_spread(lookahead_evts, octave_spread, seed=prev_seed)

    reverb_tail = 4.5 * max(decay_mult, 1.0)
    total_samp  = int((chunk_beats * beat_dur + reverb_tail) * SAMPLE_RATE) + SAMPLE_RATE
    stereo      = np.zeros((total_samp, 2), dtype=np.float32)
    rng         = np.random.default_rng(seed + 300)
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
        # Render L and R at slightly different frequencies — creates natural beating/width.
        # The absolute beat keeps a regenerated lookahead tail timbrally stable
        # across chunk boundaries while successive strikes still vary.
        strike_seed = _strike_variation_seed(beat_offset + rel_beat)
        l_tone = bell_tone(freq * (1 + _STEREO_MICRO), dur_s, velocity=vel_scaled,
                           decay_mult=decay_mult, texture=texture, attack_ms=note_atk,
                           beating=beating, strike_level=strike_level if is_normal else 0.0,
                           variation_seed=strike_seed)
        r_tone = bell_tone(freq * (1 - _STEREO_MICRO), dur_s, velocity=vel_scaled,
                           decay_mult=decay_mult, texture=texture, attack_ms=note_atk,
                           beating=beating, strike_level=strike_level if is_normal else 0.0,
                           variation_seed=strike_seed + 1)

        if is_normal and time_scatter > 0:
            grid_t     = rel_beat * beat_dur + jitter
            rand_t     = scatter_rng.uniform(0.0, chunk_beats * beat_dur)
            t          = grid_t * (1.0 - time_scatter) + rand_t * time_scatter
            start_samp = max(0, int(t * SAMPLE_RATE))
        else:
            start_samp = int((rel_beat * beat_dur + jitter) * SAMPLE_RATE)

        if start_samp < 0:
            skip = -start_samp
            if skip >= len(l_tone):
                continue
            l_tone     = l_tone[skip:]
            r_tone     = r_tone[skip:]
            start_samp = 0

        base_pan = np.clip((freq - 400) / 1200 * pan_spread, -0.5, 0.5)
        pan      = float(base_pan + rng.uniform(-0.05, 0.05))
        l_g      = np.sqrt(0.5 - pan * 0.5)
        r_g      = np.sqrt(0.5 + pan * 0.5)

        end_l  = min(start_samp + len(l_tone), total_samp)
        length = end_l - start_samp
        if length > 0:
            stereo[start_samp:end_l, 0] += l_tone[:length] * l_g
            stereo[start_samp:end_l, 1] += r_tone[:length] * r_g

    out = freeverb(stereo, room_size=reverb_room, damping=reverb_damping,
                   wet=reverb_wet, width=reverb_width, shimmer=shimmer)
    return apply_delay(out, delay_ms=delay_time, feedback=delay_feedback, wet=delay_wet)


def stereo_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Normalise and encode a stereo floating-point array to 16-bit WAV."""
    peak = max(float(np.max(audio)), -float(np.min(audio)))
    if peak > 0:
        audio *= 0.85 / peak
    audio *= 32767
    pcm = audio.astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, 'w') as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes(order='C'))
    return buf.getvalue()
