"""
engine/presets.py
-----------------
Named melody presets for the bells generator.
Each preset: dict with name, description, melody, bass, total_beats,
default_bpm, default_reverb, default_decay.
"""

# Each event: (beat_start, note_name, duration_beats, velocity 0‒1)

_MELODIC_ASCENDING_MELODY = [
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
    (47.5, 'C5',  4.5,  0.9),
]

_MELODIC_ASCENDING_BASS = [
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

_SIMPLE_ARPEGGIOS_MELODY = [
    (b * 4 + i, note, 0.8, 0.85)
    for b in range(8)
    for i, note in enumerate(['C5', 'Eb5', 'G5', 'Bb5'])
]

_SIMPLE_ARPEGGIOS_BASS = [
    (b * 4, 'C3', 3.5, 0.65) for b in range(8)
]

_MEDITATION_MELODY = [
    (0,   'G5',  3.5,  0.75),
    (4,   'Eb5', 3.5,  0.8),
    (8,   'C5',  3.5,  0.75),
    (12,  'F5',  3.5,  0.8),
    (16,  'Ab5', 3.5,  0.75),
    (20,  'G5',  3.5,  0.8),
    (24,  'Bb5', 3.5,  0.75),
    (28,  'C6',  3.5,  0.85),
    (32,  'Bb5', 3.5,  0.8),
    (36,  'Ab5', 3.5,  0.75),
    (40,  'G5',  3.5,  0.8),
    (44,  'Eb5', 3.5,  0.75),
    (48,  'C5',  8.0,  0.85),
]

_MEDITATION_BASS = [
    (0,  'C2',  7.5,  0.6),
    (8,  'F2',  7.5,  0.6),
    (16, 'Bb2', 7.5,  0.6),
    (24, 'C2',  7.5,  0.6),
    (32, 'Ab2', 7.5,  0.6),
    (40, 'C2',  7.5,  0.6),
    (48, 'G2',  8.0,  0.6),
]

_CASCADING_MELODY = []
patterns = [
    ['C6', 'Bb5', 'Ab5', 'G5', 'F5', 'Eb5', 'D5', 'C5'],
    ['Bb5', 'Ab5', 'G5', 'F5', 'Eb5', 'D5', 'C5', 'Bb4'],
    ['G5', 'F5', 'Eb5', 'D5', 'C5', 'Bb4', 'Ab4', 'G4'],
    ['F5', 'Eb5', 'D5', 'C5', 'Bb4', 'Ab4', 'G4', 'F4'],
]
for b, pat in enumerate(patterns):
    for i, note in enumerate(pat):
        _CASCADING_MELODY.append((b * 8 + i * 1.0, note, 0.9, 0.8 + 0.05 * (i % 2)))
# second half ascending mirror
_CASCADING_MELODY += [
    (32 + i * 0.75, note, 0.65, 0.85)
    for i, note in enumerate(['C5','Eb5','G5','Bb5','C6','Bb5','Ab5','G5',
                               'F5','G5','Eb5','F5','D5','Eb5','C5','D5'])
]

_CASCADING_BASS = [(b * 4, n, 3.5, 0.65) for b, n in enumerate(
    ['C3','Bb2','Ab2','G2','F2','Eb2','G2','C3','F2','C3']
)]

_EVENING_BELLS_MELODY = [
    (0,   'C5',  4.0, 0.9),
    (4,   'Eb5', 4.0, 0.85),
    (8,   'G5',  4.0, 0.9),
    (12,  'C6',  4.0, 0.95),
    (16,  'Bb5', 4.0, 0.88),
    (20,  'G5',  4.0, 0.85),
    (24,  'F5',  4.0, 0.88),
    (28,  'Eb5', 4.0, 0.9),
    (32,  'G5',  4.0, 0.85),
    (36,  'Ab5', 4.0, 0.88),
    (40,  'Bb5', 4.0, 0.9),
    (44,  'C6',  4.0, 0.92),
    (48,  'Bb5', 2.0, 0.88),
    (50,  'G5',  2.0, 0.85),
    (52,  'Eb5', 2.0, 0.88),
    (54,  'C5',  6.0, 0.95),
]

_EVENING_BELLS_BASS = [
    (b * 8, n, 7.5, 0.6)
    for b, n in enumerate(['C2','F2','Eb2','Bb2','Ab2','C2','F2','C2'])
]

PRESETS = {
    "meditation": {
        "name":            "Meditation Tones",
        "description":     "Very slow, widely spaced single tones for meditation",
        "melody":          _MEDITATION_MELODY,
        "bass":            _MEDITATION_BASS,
        "total_beats":     60,
        "default_bpm":     50,
        "default_reverb":  0.75,
        "default_decay":   1.4,
    },
    "evening_bells": {
        "name":            "Evening Bells",
        "description":     "Slow tolling church-bell style, wide spacing",
        "melody":          _EVENING_BELLS_MELODY,
        "bass":            _EVENING_BELLS_BASS,
        "total_beats":     64,
        "default_bpm":     45,
        "default_reverb":  0.8,
        "default_decay":   1.6,
    },
}

KEYS = {
    "C minor":  0,
    "D minor":  2,
    "Eb minor": 3,
    "E minor":  4,
    "F minor":  5,
    "G minor":  7,
    "A minor":  9,
    "Bb minor": 10,
}
