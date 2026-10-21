"""Check eager JS bindings against real markup, without inventing missing nodes."""
import re
import unittest
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MarkupIds(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.ids = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.ids.extend(value for name, value in attrs if name == 'id')


class FrontendMarkupTests(unittest.TestCase):
    def test_every_eager_ui_binding_exists_in_real_page(self):
        markup = MarkupIds((ROOT / 'static/index.html').read_text(encoding='utf-8'))
        script = (ROOT / 'static/app.js').read_text(encoding='utf-8')
        bindings = script.split('  const ids = [', 1)[1].split('  const ui = Object.fromEntries', 1)[0]
        referenced = set(re.findall(r'"([^"\n]+)"', bindings))
        self.assertGreater(len(referenced), 100, 'Expected the real eager binding list.')
        self.assertEqual(referenced - set(markup.ids), set(), 'Missing startup UI elements')
        self.assertEqual(len(markup.ids), len(set(markup.ids)), 'Duplicate HTML IDs')

    def test_cinematic_controls_exist_in_real_page(self):
        markup = MarkupIds((ROOT / 'static/index.html').read_text(encoding='utf-8'))
        script = (ROOT / 'static/studio.js').read_text(encoding='utf-8')
        referenced = set(re.findall(r"\$\('([^']+)'\)", script))
        self.assertGreater(len(referenced), 15)
        self.assertEqual(referenced - set(markup.ids), set(), 'Missing cinematic controls')


if __name__ == '__main__':
    unittest.main()
