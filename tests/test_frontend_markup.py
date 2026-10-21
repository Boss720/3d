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


class MarkupTree(HTMLParser):
    VOID_TAGS = {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input', 'link', 'meta', 'param', 'source', 'track', 'wbr'}

    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.stack = []
        self.feed(html)

    def _start(self, tag, attrs, void=False):
        element = {'tag': tag, 'attrs': dict(attrs), 'parent': self.stack[-1] if self.stack else None}
        self.elements.append(element)
        if not void and tag not in self.VOID_TAGS:
            self.stack.append(element)

    def handle_starttag(self, tag, attrs):
        self._start(tag, attrs)

    def handle_startendtag(self, tag, attrs):
        self._start(tag, attrs, void=True)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index]['tag'] == tag:
                del self.stack[index:]
                return


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

    def test_live_preview_and_queue_statuses_stay_outside_disclosures(self):
        tree = MarkupTree((ROOT / 'static/index.html').read_text(encoding='utf-8'))
        by_id = {node['attrs'].get('id'): node for node in tree.elements if node['attrs'].get('id')}
        for node_id in ('previewState', 'queueLiveStatus'):
            node = by_id[node_id]
            self.assertEqual(node['attrs'].get('role'), 'status')
            self.assertEqual(node['attrs'].get('aria-live'), 'polite')
            ancestors = []
            parent = node['parent']
            while parent:
                ancestors.append(parent)
                parent = parent['parent']
            self.assertFalse(any(parent['tag'] == 'details' for parent in ancestors), f'{node_id} must remain available while disclosures are closed')
        self.assertEqual(by_id['queueLiveStatus']['attrs'].get('aria-atomic'), 'true')

    def test_import_uses_one_native_keyboard_control(self):
        tree = MarkupTree((ROOT / 'static/index.html').read_text(encoding='utf-8'))
        by_id = {node['attrs'].get('id'): node for node in tree.elements if node['attrs'].get('id')}
        drop_zone = by_id['dropZone']
        self.assertNotIn('role', drop_zone['attrs'])
        self.assertNotIn('tabindex', drop_zone['attrs'])
        browse = by_id['browseButton']
        self.assertEqual(browse['tag'], 'button')
        self.assertIs(browse['parent'], drop_zone)

    def test_about_dialog_name_resolves_to_its_heading(self):
        tree = MarkupTree((ROOT / 'static/index.html').read_text(encoding='utf-8'))
        by_id = {node['attrs'].get('id'): node for node in tree.elements if node['attrs'].get('id')}
        dialog = by_id['aboutDialog']
        heading_id = dialog['attrs'].get('aria-labelledby')
        self.assertIsNotNone(heading_id)
        self.assertEqual(by_id[heading_id]['tag'], 'h2')


if __name__ == '__main__':
    unittest.main()
