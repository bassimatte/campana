import re
import unittest
from pathlib import Path


class AnalyticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.static_html = Path("engine/static/index.html").read_text(encoding="utf-8-sig")
        cls.docs_html = Path("docs/index.html").read_text(encoding="utf-8-sig")

    def test_deployed_and_local_interfaces_differ_only_by_api_helper(self):
        docs = self.docs_html.replace(
            "const CAMPANA_API_BASE = 'https://campana-production.up.railway.app';\n"
            "    function apiUrl(path) { return CAMPANA_API_BASE + path; }",
            "function apiUrl(path) { return path; }",
        )
        self.assertEqual(self.static_html, docs)

    def test_analytics_is_limited_to_canonical_campana_path(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn("const CAMPANA_ANALYTICS_HOST = 'bassimatte.github.io';", html)
            self.assertIn("const CAMPANA_ANALYTICS_PATH = '/campana';", html)
            self.assertIn(
                "location.pathname.startsWith(`${CAMPANA_ANALYTICS_PATH}/`)", html
            )
            self.assertIn("script.dataset.domains = CAMPANA_ANALYTICS_HOST;", html)

    def test_shared_website_is_separated_by_campana_tag(self):
        for html in (self.static_html, self.docs_html):
            match = re.search(r"const CAMPANA_UMAMI_WEBSITE_ID = '([^']*)';", html)
            self.assertIsNotNone(match)
            website_id = match.group(1)
            self.assertRegex(
                website_id,
                r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
                r"[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
            )
            self.assertIn("const CAMPANA_ANALYTICS_TAG = 'campana';", html)
            self.assertIn("script.dataset.tag = CAMPANA_ANALYTICS_TAG;", html)

    def test_umami_privacy_controls_are_enabled(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn("script.dataset.excludeSearch = 'true';", html)
            self.assertIn("script.dataset.excludeHash = 'true';", html)
            self.assertIn("script.dataset.doNotTrack = 'true';", html)
            self.assertIn("if (!analyticsIsConfigured()) return false;", html)

    def test_events_and_properties_are_allowlisted(self):
        schema_match = re.search(
            r"const CAMPANA_ANALYTICS_SCHEMA = Object\.freeze\(\{(.*?)\n\}\);",
            self.static_html,
            re.DOTALL,
        )
        self.assertIsNotNone(schema_match)
        schema = schema_match.group(1)
        for event in (
            "campana_audio_started",
            "campana_audio_failed",
            "campana_listening_reached",
            "campana_export_completed",
            "campana_export_failed",
        ):
            self.assertIn(f"{event}:", schema)
            self.assertRegex(self.static_html, rf"trackUsage\('{event}'")

        self.assertIn("if (allowed.includes(value)) props[key] = value;", self.static_html)
        self.assertIn("window.umami.track(eventName, properties);", self.static_html)

    def test_public_festa_name_is_reported(self):
        self.assertIn("'festa'", self.static_html)
        self.assertIn("value === 'giardino' ? 'festa'", self.static_html)
        campana_audio_started_schema = re.search(
            r"campana_audio_started: \{(.*?)\n  \},", self.static_html, re.DOTALL
        ).group(1)
        self.assertNotIn("'giardino'", campana_audio_started_schema)

    def test_sensitive_dynamic_values_are_not_sent(self):
        calls = re.findall(
            r"trackUsage\('([^']+)'\s*,\s*\{(.*?)\}\);",
            self.static_html,
            re.DOTALL,
        )
        self.assertTrue(calls)
        property_text = "\n".join(properties for _, properties in calls)
        for forbidden_key in (
            "seed:",
            "value:",
            "filename:",
            "audio:",
            "error:",
            "message:",
            "query:",
        ):
            self.assertNotIn(forbidden_key, property_text)

    def test_privacy_notice_is_visible(self):
        for html in (self.static_html, self.docs_html):
            self.assertIn("anonymous, aggregate usage statistics", html)
            self.assertIn("Analytics is disabled for local installations", html)


if __name__ == "__main__":
    unittest.main()
