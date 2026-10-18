import unittest
from unittest.mock import patch

from gpu_hardware import probe_hardware, select_adapter


class HardwareTest(unittest.TestCase):
    def test_nvidia_uses_dxgi_index_not_nvidia_smi_index(self):
        adapters = [
            {'id': 0, 'name': 'Intel UHD', 'vendor_id': 0x8086, 'memory_mb': 128, 'software': False},
            {'id': 1, 'name': 'NVIDIA GTX 1650 Ti', 'vendor_id': 0x10DE, 'memory_mb': 4096, 'software': False},
        ]
        self.assertEqual(select_adapter(adapters)['id'], 1)

    def test_highest_memory_nvidia_preferred_and_software_excluded(self):
        adapters = [
            {'id': 0, 'name': 'NVIDIA small', 'vendor_id': 0x10DE, 'memory_mb': 2048},
            {'id': 1, 'name': 'AMD big', 'vendor_id': 0x1002, 'memory_mb': 16384},
            {'id': 2, 'name': 'NVIDIA big', 'vendor_id': 0x10DE, 'memory_mb': 4096},
            {'id': 3, 'name': 'Software', 'vendor_id': 0x10DE, 'memory_mb': 99999, 'software': True},
        ]
        self.assertEqual(select_adapter(adapters)['id'], 2)
        self.assertEqual(select_adapter([adapters[1]])['id'], 1)
        self.assertIsNone(select_adapter([adapters[3]]))

    def test_probe_is_cached_and_preserves_enumeration_errors(self):
        probe_hardware.cache_clear()
        self.addCleanup(probe_hardware.cache_clear)
        with patch('gpu_hardware.enumerate_adapters', side_effect=RuntimeError('DXGI unavailable')) as enumerate_:
            first = probe_hardware()
            second = probe_hardware()
            self.assertEqual(first, second)
            self.assertEqual(enumerate_.call_count, 1)
            self.assertIn('DXGI unavailable', first['error'])
            self.assertEqual(first['adapters'], [])


if __name__ == '__main__':
    unittest.main()
