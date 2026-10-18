import unittest
from datetime import datetime, timedelta, timezone

from output_naming import DEFAULT_OUTPUT_NAME_TEMPLATE, render_output_name, validate_output_template


JOB_ID = '12345678-1111-4000-8000-000000000001'
SUFFIX = '12345678111140008000000000000001'


class OutputNamingTests(unittest.TestCase):
    def render(self, template=DEFAULT_OUTPUT_NAME_TEMPLATE, **changes):
        values = dict(filename='source.png', layout='SBS', model='depth-anything-v2-small',
                      created_at=datetime(2026, 10, 18, 0, 30, tzinfo=timezone.utc).timestamp(),
                      job_id=JOB_ID, media_type='image')
        values.update(changes)
        return render_output_name(template=template, **values)

    def test_default_matches_ordinary_existing_basename_and_full_identity(self):
        self.assertEqual(self.render(), f'source_sbs_{SUFFIX}.png')
        self.assertEqual(self.render(media_type='video'), f'source_sbs_{SUFFIX}.mp4')

    def test_only_named_tokens_expand_and_values_are_not_recursively_formatted(self):
        self.assertEqual(self.render('{name}-{layout}-{model}-{date}'),
                         f'source-sbs-depth-anything-v2-small-20261018_{SUFFIX}.png')
        self.assertEqual(self.render('{name}-{date}', filename='{model}.png'),
                         f'model_-20261018_{SUFFIX}.png')

    def test_strict_parser_rejects_unsupported_formatting_paths_controls_and_literal_characters(self):
        invalid = ('', '   ', 'x' * 121, '{unknown}', '{}', '{name', 'name}', '{{name}}', '{name.x}',
                   '{name[0]}', '{name!r}', '{date:%Y}', '{name:>10}', '{name!s:>2}',
                   '../{name}', r'folder\{name}', 'C:{name}', '<name>', 'a"b', 'a|b', 'a?b', 'a*b',
                   'a\0b', 'a\nb', 'a\rb', 'a\tb', 'a\x7fb', 'a\x85b', 'a\u202eb', 'a\u200bb', None, 1, True, [], {})
        for template in invalid:
            with self.subTest(template=template), self.assertRaises(ValueError):
                validate_output_template(template)

    def test_supported_unicode_literals_and_repeated_tokens_are_valid(self):
        for template in ('{name}', '{date}_{date}', 'Scene 01 - {name}.final', '电影_{layout}', 'x' * 120, '...'):
            with self.subTest(template=template):
                self.assertEqual(validate_output_template(template), template)

    def test_source_directories_and_windows_invalid_source_characters_never_escape_basename(self):
        name = self.render('{name}', filename=r'C:\private\folder\evil:?*name.png')
        self.assertEqual(name, f'evil_name_{SUFFIX}.png')
        self.assertNotIn('private', name)
        self.assertEqual(self.render('{name}', filename='../../hidden/source.png'), f'source_{SUFFIX}.png')

    def test_entire_rendered_stem_is_ascii_bounded_and_empty_falls_back(self):
        name = self.render('{name}_{model}_{date}', filename='a' * 300 + '.png')
        self.assertEqual(name, 'a' * 100 + f'_{SUFFIX}.png')
        self.assertEqual(len(name), 137)
        self.assertEqual(self.render('电影'), f'stereo-output_{SUFFIX}.png')
        self.assertEqual(self.render('...'), f'stereo-output_{SUFFIX}.png')
        self.assertTrue(self.render('电影_{layout}').isascii())

    def test_date_is_stable_created_timestamp_in_utc_across_timezone_boundary(self):
        stamp = datetime(2026, 10, 18, 0, 30, tzinfo=timezone(timedelta(hours=2))).timestamp()
        expected = f'20261017_{SUFFIX}.png'
        self.assertEqual(self.render('{date}', created_at=stamp), expected)
        self.assertEqual(self.render('{date}', created_at=stamp), expected)

    def test_same_prefix_distinct_full_ids_do_not_collide_after_stem_truncation(self):
        template = 'x' * 120
        first = self.render(template)
        second = self.render(template, job_id='12345678-1111-4000-8000-000000000002')
        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith(f'_{SUFFIX}.png'))

    def test_non_uuid_internal_ids_have_stable_safe_32_character_identity(self):
        first = self.render(job_id='test/internal/id')
        self.assertEqual(first, self.render(job_id='test/internal/id'))
        identity = first.removesuffix('.png').rsplit('_', 1)[-1]
        self.assertEqual(len(identity), 32)
        self.assertTrue(all(character in '0123456789abcdef' for character in identity))

    def test_windows_device_names_before_literal_dot_are_safe(self):
        for name in ('CON.foo', 'aux.final', 'LPT1.final', 'COM9 .final'):
            with self.subTest(name=name):
                self.assertEqual(self.render(name), f'_{name}_{SUFFIX}.png')

    def test_unpaired_surrogates_are_invalid_but_astral_unicode_is_valid(self):
        with self.assertRaises(ValueError):
            validate_output_template('\ud800_{name}')
        self.assertEqual(validate_output_template('\U0001f680_{name}'), '\U0001f680_{name}')
