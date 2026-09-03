import re
import unittest
from pathlib import Path

import numpy as np

from engine.presets import PRESETS, resolve_preset_id
from engine.synth import (
    BELL_TEXTURES,
    _strike_variation_seed,
    bell_tone,
    generate_bell_events,
)
from engine.web_server import _parse_render_params


class PresetIdentityTests(unittest.TestCase):
    def test_every_preset_default_bpm_is_available_in_the_interface(self):
        expected = {str(preset["default_bpm"]) for preset in PRESETS.values()}
        for path in ("engine/static/index.html", "docs/index.html"):
            html = Path(path).read_text(encoding="utf-8-sig")
            select = re.search(
                r'<select id="bpmSelect">(.*?)</select>', html, re.DOTALL
            )
            self.assertIsNotNone(select)
            available = set(
                re.findall(r'<option value="([^"]+)"', select.group(1))
            )
            self.assertEqual(expected - available, set(), path)

    def test_festa_is_canonical_and_giardino_is_only_an_alias(self):
        self.assertIn("festa", PRESETS)
        self.assertNotIn("giardino", PRESETS)
        self.assertEqual(resolve_preset_id("festa"), "festa")
        self.assertEqual(resolve_preset_id("giardino"), "festa")
        self.assertEqual(_parse_render_params({"preset": "giardino"})["preset_id"], "festa")

        for path in ("engine/static/index.html", "docs/index.html"):
            html = Path(path).read_text(encoding="utf-8-sig")
            self.assertIn("selectPreset('festa')", html)
            self.assertNotIn("selectPreset('giardino')", html)
            self.assertIn("id:'festa'", html)
            self.assertNotIn("id:'giardino'", html)
            self.assertIn('data-val="bronze"', html)
            self.assertIn('data-val="handbell"', html)

    def test_pair_uses_dedicated_generators_and_textures(self):
        cattedrale = PRESETS["cattedrale"]
        festa = PRESETS["festa"]
        self.assertEqual(cattedrale["gen_params"]["style"], "tolling")
        self.assertEqual(cattedrale["default_texture"], "bronze")
        self.assertEqual(festa["gen_params"]["style"], "carillon")
        self.assertEqual(festa["default_texture"], "handbell")
        self.assertGreater(len(BELL_TEXTURES["bronze"]), 5)
        self.assertGreater(len(BELL_TEXTURES["handbell"]), 5)
        self.assertLess(BELL_TEXTURES["bronze"][0][0], 1.0)

    def test_specialized_generators_are_deterministic(self):
        for preset_id in ("cattedrale", "festa"):
            preset = PRESETS[preset_id]
            first = generate_bell_events(
                preset["default_scale_mode"], 16, preset["gen_params"], 2026
            )
            second = generate_bell_events(
                preset["default_scale_mode"], 16, preset["gen_params"], 2026
            )
            self.assertEqual(first, second)

    def test_cattedrale_and_festa_have_measurably_different_gestures(self):
        cattedrale = PRESETS["cattedrale"]
        festa = PRESETS["festa"]
        tolls = generate_bell_events(
            cattedrale["default_scale_mode"], 16, cattedrale["gen_params"], 42
        )
        carillon = generate_bell_events(
            festa["default_scale_mode"], 16, festa["gen_params"], 42
        )

        self.assertLessEqual(len(tolls), 6)
        self.assertGreaterEqual(len(carillon), len(tolls) * 6)

        toll_octaves = [int(note[-1]) for _, note, _, _ in tolls]
        carillon_octaves = [int(note[-1]) for _, note, _, _ in carillon]
        self.assertLess(np.median(toll_octaves), np.median(carillon_octaves))

        toll_beats = sorted({round(event[0], 4) for event in tolls})
        carillon_beats = sorted({round(event[0], 4) for event in carillon})
        toll_gaps = np.diff(toll_beats)
        carillon_gaps = np.diff(carillon_beats)
        self.assertGreater(np.median(toll_gaps), 2.0)
        self.assertLessEqual(np.max(carillon_gaps), 0.5)

    def test_each_strike_can_have_deterministic_timbre_variation(self):
        first = bell_tone(440.0, 0.08, texture="handbell", variation_seed=1)
        repeated = bell_tone(440.0, 0.08, texture="handbell", variation_seed=1)
        varied = bell_tone(440.0, 0.08, texture="handbell", variation_seed=2)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first, varied))

    def test_strike_timbre_stays_stable_when_tail_crosses_a_chunk(self):
        initial_chunk_seed = _strike_variation_seed(0.0 + 6.5)
        lookahead_chunk_seed = _strike_variation_seed(12.0 - 5.5)
        self.assertEqual(initial_chunk_seed, lookahead_chunk_seed)


if __name__ == "__main__":
    unittest.main()
