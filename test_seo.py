import re
import json
import struct
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
                self.assertEqual(
                    data["author"],
                    {
                        "@type": "Person",
                        "@id": "https://bassimatte.github.io/#person",
                        "name": "Matteo Bassi",
                        "url": "https://bassimatte.github.io/",
                        "sameAs": [
                            "https://github.com/bassimatte",
                            "https://freesound.org/people/bassimat/",
                        ],
                    },
                )
                self.assertEqual(data["offers"]["price"], "0")

    def test_open_graph_metadata_has_a_public_social_card(self):
        expected = {
            "og:type": "website",
            "og:site_name": "Campana",
            "og:title": "Campana – Generative Bell Synthesizer",
            "og:url": "https://bassimatte.github.io/campana/",
            "og:image": "https://bassimatte.github.io/campana/social-card.png",
            "og:image:type": "image/png",
            "og:image:width": "2400",
            "og:image:height": "2400",
        }
        for path, html in self.targets.items():
            with self.subTest(path=path):
                metadata = dict(
                    re.findall(
                        r'<meta property="(og:[^"]+)" content="([^"]+)"\s*/>', html
                    )
                )
                self.assertEqual({key: metadata.get(key) for key in expected}, expected)
                self.assertTrue(metadata.get("og:description"))
                self.assertTrue(metadata.get("og:image:alt"))

    def test_twitter_metadata_matches_the_square_card(self):
        expected_image = "https://bassimatte.github.io/campana/social-card.png"
        for path, html in self.targets.items():
            with self.subTest(path=path):
                metadata = dict(
                    re.findall(
                        r'<meta name="(twitter:[^"]+)" content="([^"]+)"\s*/>', html
                    )
                )
                self.assertEqual(metadata.get("twitter:card"), "summary")
                self.assertEqual(metadata.get("twitter:image"), expected_image)
                self.assertTrue(metadata.get("twitter:title"))
                self.assertTrue(metadata.get("twitter:description"))
                self.assertTrue(metadata.get("twitter:image:alt"))

    def test_social_card_is_in_the_github_pages_output(self):
        card = Path("docs/social-card.png")
        self.assertTrue(card.is_file())
        self.assertEqual(card.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_creator_links_back_to_personal_site(self):
        for path, html in self.targets.items():
            with self.subTest(path=path):
                self.assertIn(
                    '<a class="app-byline creator-link" '
                    'href="https://bassimatte.github.io/">by Matteo Bassi</a>',
                    html,
                )
                self.assertIn(
                    '<a class="creator-link" '
                    'href="https://bassimatte.github.io/#instruments">'
                    'More tools by Matteo Bassi ↗</a>',
                    html,
                )
                personal_links = re.findall(
                    r'<a[^>]+href="https://bassimatte\.github\.io/(?:#instruments)?"[^>]*>',
                    html,
                )
                self.assertEqual(len(personal_links), 2)
                self.assertTrue(all("nofollow" not in link for link in personal_links))
                self.assertIn(
                    ".app-byline.creator-link {\n      color: var(--text);\n    }",
                    html,
                )

    def test_favicon_is_declared_and_deployed_in_both_apps(self):
        expected_links = (
            '<link rel="icon" type="image/png" sizes="512x512" '
            'href="favicon.png" />',
            '<link rel="apple-touch-icon" href="favicon.png" />',
        )
        for path, html in self.targets.items():
            with self.subTest(path=path):
                for link in expected_links:
                    self.assertIn(link, html)

        icon_paths = (
            Path("engine/static/favicon.png"),
            Path("docs/favicon.png"),
        )
        icons = [path.read_bytes() for path in icon_paths]
        self.assertEqual(icons[0], icons[1])
        self.assertEqual(icons[0][:8], b"\x89PNG\r\n\x1a\n")
        width, height = struct.unpack(">II", icons[0][16:24])
        self.assertEqual((width, height), (512, 512))


if __name__ == "__main__":
    unittest.main()
