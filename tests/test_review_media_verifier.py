"""A lossy hardware encode cannot promise byte identity; audio still must match."""
import copy
import unittest
from unittest.mock import patch

from scripts.verify_packaged_release import compare_repeat_media


class RepeatMediaVerifierTests(unittest.TestCase):
    def records(self):
        first = {'path': 'first.mp4', 'encoded_sha256': 'a', 'decoded_video_sha256': 'a',
                 'decoded_video_bytes': 36, 'decoded_audio_sha256': 'same', 'decoded_audio_bytes': 8,
                 'probe': {'streams': [{'codec_type': 'video', 'width': 3, 'height': 2,
                                        'nb_read_frames': '2', 'avg_frame_rate': '2/1',
                                        'start_time': '0', 'duration': '1'},
                                       {'codec_type': 'audio', 'sample_rate': '48000', 'channels': 1,
                                        'start_time': '0', 'duration': '1', 'channel_layout': 'mono'}],
                           'format': {'duration': '1'}}}
        second = copy.deepcopy(first)
        second.update(path='repeat.mp4', encoded_sha256='b', decoded_video_sha256='b')
        return first, second

    def test_variation_is_explicit_and_does_not_claim_identity(self):
        first, second = self.records()
        with self.assertRaisesRegex(AssertionError, 'not enabled'):
            compare_repeat_media('ffmpeg', first, second)
        repeat = bytearray(36)
        repeat[0] = 1
        with patch('scripts.verify_packaged_release.ffmpeg_run', side_effect=[bytes(36), bytes(repeat)]):
            result = compare_repeat_media('ffmpeg', first, second, allow_hardware_variation=True)
        self.assertFalse(result['encoded_identical'])
        self.assertFalse(result['decoded_video_identical'])
        self.assertTrue(result['decoded_audio_identical'])

    def test_changed_frame_is_rejected_individually(self):
        first, second = self.records()
        with patch('scripts.verify_packaged_release.ffmpeg_run', side_effect=[bytes(36), bytes(18) + bytes([2]) * 18]):
            with self.assertRaisesRegex(AssertionError, 'frame 1 exceeds'):
                compare_repeat_media('ffmpeg', first, second, allow_hardware_variation=True)

    def test_audio_and_geometry_remain_exact(self):
        for change in ('audio', 'geometry', 'audio_timing', 'container_timing'):
            first, second = self.records()
            if change == 'audio':
                second['decoded_audio_sha256'] = 'changed'
            elif change == 'geometry':
                second['probe']['streams'][0]['width'] = 4
            elif change == 'audio_timing':
                second['probe']['streams'][1]['start_time'] = '0.25'
            else:
                second['probe']['format']['duration'] = '1.25'
            with self.assertRaises(AssertionError):
                compare_repeat_media('ffmpeg', first, second, allow_hardware_variation=True)
