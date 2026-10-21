"""Source and frozen releases must share an exact, versioned code identity."""
import json
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import main


class ReleaseIdentityTests(unittest.TestCase):
    def test_frozen_manifest_matches_source_and_invalidates_after_change(self):
        from build_identity import SOURCE_COMPONENTS, build_manifest, build_fingerprint
        root = main.RUNTIME / 'release-identity-tests' / uuid.uuid4().hex
        root.mkdir(parents=True)
        for name in SOURCE_COMPONENTS:
            (root / name).write_text(name, encoding='utf-8')
        manifest = build_manifest(root)
        (root / 'stereolab-build.json').write_text(json.dumps(manifest), encoding='utf-8')
        self.assertEqual(build_fingerprint(root), build_fingerprint(root, frozen=True))
        (root / 'stereo.py').write_text('changed algorithm', encoding='utf-8')
        self.assertNotEqual(build_fingerprint(root), build_fingerprint(root, frozen=True))

    def test_missing_or_invalid_frozen_manifest_has_no_identity(self):
        from build_identity import build_fingerprint
        root = main.RUNTIME / 'release-identity-tests' / uuid.uuid4().hex
        root.mkdir(parents=True)
        self.assertIsNone(build_fingerprint(root, frozen=True))
        for content in ('{}', 'not json', '{"schema_version": 1, "components": {}}'):
            (root / 'stereolab-build.json').write_text(content, encoding='utf-8')
            self.assertIsNone(build_fingerprint(root, frozen=True))

    def test_unknown_build_cannot_read_or_create_depth_cache(self):
        job = main.Job(id='unknown-build', filename='test.mp4', input_path=Path('none'),
                       media_type='video', layout='SBS', strength=4, convergence=.5,
                       depth_mode='depth-anything-v2-small', input_size=256,
                       resolution='source', quality='fast', invert_depth=False,
                       execution_mode='cpu', use_depth_cache=True)
        with patch.object(main, 'APP_BUILD_FINGERPRINT', None), \
                patch.object(main, 'get_video_depth_store') as store:
            session = main.VideoDepthCacheSession(job)
            session.start(job.input_path, 32, 24, 12, {})
        store.assert_not_called()
        self.assertEqual(job.depth_cache['state'], 'unavailable')
        self.assertIn('build identity', job.depth_cache['warning'])

    def test_launcher_port_is_bounded_and_service_stays_loopback(self):
        from app_launcher import run_app
        with patch('app_launcher.uvicorn.run') as run:
            run_app(main.app, ['--port', '18765'])
            run.assert_called_once_with(main.app, host='127.0.0.1', port=18765, log_level='warning')
        for value in ('0', '65536', 'text'):
            with self.subTest(value=value), patch('app_launcher.uvicorn.run') as run:
                with self.assertRaises(SystemExit) as error:
                    run_app(main.app, ['--port', value])
                self.assertEqual(error.exception.code, 2)
                run.assert_not_called()
