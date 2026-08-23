import re
import json
import unittest
from pathlib import Path


class SeoMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.targets = {
            path: Path(path).read_text(encoding="utf-8-sig")
            for path in ("engine/static/index.html", "docs/index.html")
        }

    def test_title_describes_the_product(self):
        expected = "Campana – Generative Bell Synthesizer &amp; Ambient Soundscapes"
        for path, html in self.targets.items():
            with self.subTest(path=path):
                self.assertEqual(re.findall(r"<title>(.*?)</title>", html), [expected])

    def test_meta_description_is_unique_and_usefully_sized(self):
        for path, html in self.targets.items():
            with self.subTest(path=path):
                descriptions = re.findall(
                    r'<meta name="description" content="([^"]+)"\s*/>', html
                )
                self.assertEqual(len(descriptions), 1)
                self.assertIn("generative synthesizer", descriptions[0])
                self.assertIn("bell soundscapes", descriptions[0])
                self.assertGreaterEqual(len(descriptions[0]), 120)
                self.assertLessEqual(len(descriptions[0]), 170)

    def test_canonical_url_points_to_the_public_app(self):
        expected = "https://bassimatte.github.io/campana/"
        for path, html in self.targets.items():
            with self.subTest(path=path):
                canonicals = re.findall(
                    r'<link rel="canonical" href="([^"]+)"\s*/>', html
                )
                self.assertEqual(canonicals, [expected])

    def test_page_has_one_descriptive_h1(self):
        for path, html in self.targets.items():
            with self.subTest(path=path):
                self.assertEqual(re.findall(r"<h1[^>]*>(.*?)</h1>", html), ["Campana"])
                self.assertIn(
                    "Generative bell synthesizer &amp; ambient soundscapes", html
                )

    def test_structured_data_describes_a_free_web_application(self):
        for path, html in self.targets.items():
            with self.subTest(path=path):
                blocks = re.findall(
                    r'<script type="application/ld\+json">\s*(.*?)\s*</script>',
                    html,
                    re.DOTALL,
                )
                self.assertEqual(len(blocks), 1)
                data = json.loads(blocks[0])
                self.assertEqual(data["@context"], "https://schema.org")
                self.assertEqual(data["@type"], "WebApplication")
                self.assertEqual(data["name"], "Campana")
                self.assertEqual(data["url"], "https://bassimatte.github.io/campana/")
                self.assertEqual(data["offers"]["price"], "0")


if __name__ == "__main__":
    unittest.main()
