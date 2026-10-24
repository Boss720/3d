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
        with patch('gpu_hardware.sys.platform', 'win32'), patch('gpu_hardware.enumerate_adapters', side_effect=RuntimeError('DXGI unavailable')) as enumerate_:
            first = probe_hardware()
            second = probe_hardware()
            self.assertEqual(first, second)
            self.assertEqual(enumerate_.call_count, 1)
            self.assertIn('DXGI unavailable', first['error'])
            self.assertEqual(first['adapters'], [])

    def test_linux_cpu_probe_does_not_enumerate_dxgi(self):
        probe_hardware.cache_clear()
        self.addCleanup(probe_hardware.cache_clear)
        with patch('gpu_hardware.sys.platform', 'linux'), patch('gpu_hardware.enumerate_adapters') as enumerate_:
            result = probe_hardware()
        enumerate_.assert_not_called()
        self.assertEqual(result, {'adapters': [], 'selected_adapter': None, 'error': None})



class CudaInventoryTest(unittest.TestCase):
    def setUp(self):
        import gpu_hardware
        from unittest.mock import Mock
        self.hardware = gpu_hardware
        self.hardware.probe_cuda_devices.cache_clear()
        self.addCleanup(self.hardware.probe_cuda_devices.cache_clear)
        self.devices = [dict(name='NVIDIA Turing', memory=4096, cc=(7, 5), pci='0000:02:00.0', uuid=bytes(range(16)))]
        self.driver = Mock()
        def count(out): out._obj.value = len(self.devices); return 0
        def version(out): out._obj.value = 12090; return 0
        def properties(out, ordinal):
            prop = out._obj
            prop.name = self.devices[ordinal]['name'].encode()
            prop.memory = self.devices[ordinal]['memory'] * 1024**2
            prop.uuid[:] = self.devices[ordinal]['uuid']
            return 0
        def attribute(out, attr, ordinal): out._obj.value = self.devices[ordinal]['cc'][attr - 75]; return 0
        def pci(out, size, ordinal): out.value = self.devices[ordinal]['pci'].encode(); return 0
        for method, effect in {'cudaRuntimeGetVersion': version, 'cudaGetDeviceCount': count,
                               'cudaGetDeviceProperties_v2': properties, 'cudaDeviceGetAttribute': attribute,
                               'cudaDeviceGetPCIBusId': pci}.items():
            getattr(self.driver, method).side_effect = effect
        self.real_loader = self.hardware._load_cuda_runtime
        self.loader = patch('gpu_hardware._load_cuda_runtime', return_value=self.driver)
        self.loader.start(); self.addCleanup(self.loader.stop)
        self.platform = patch('gpu_hardware.sys.platform', 'win32')
        self.platform.start(); self.addCleanup(self.platform.stop)

    def probe(self, output='0, NVIDIA other, 8192, TCC, 00000000:01:00.0\n1, NVIDIA Turing, 4096, WDDM, 00000000:02:00.0\n'):
        from types import SimpleNamespace
        with patch('gpu_hardware.subprocess.run', return_value=SimpleNamespace(stdout=output)):
            return self.hardware.probe_cuda_devices()

    def test_visible_cuda_ordinal_not_physical_index_and_capability(self):
        import os
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}):
            result = self.probe()
        self.assertIsNone(result['error'])
        self.assertEqual(len(result['devices']), 1)
        device = result['devices'][0]
        self.assertEqual(device['id'], 0)
        self.assertEqual(device['name'], 'NVIDIA Turing')
        self.assertEqual(device['memory_mb'], 4096)
        self.assertEqual(device['compute_capability'], [7, 5])
        self.assertEqual(device['driver_model'], 'WDDM')
        self.driver.cudaGetDeviceCount.assert_called_once()

    def test_reordered_devices_keep_cuda_driver_order(self):
        self.devices.append(dict(name='NVIDIA other', memory=8192, cc=(8, 0), pci='0000:01:00.0', uuid=bytes(range(1,17))))
        result = self.probe()['devices']
        self.assertEqual([(d['id'], d['name'], d['driver_model']) for d in result], [(0,'NVIDIA Turing','WDDM'),(1,'NVIDIA other','TCC')])

    def test_missing_smi_does_not_disable_working_driver(self):
        with patch('gpu_hardware.subprocess.run', side_effect=FileNotFoundError('no CLI')):
            result = self.hardware.probe_cuda_devices()
        self.assertIsNone(result['error'])
        self.assertEqual(result['devices'][0]['compute_capability'], [7,5])

    def test_driver_failure_never_falls_back_to_physical_indices(self):
        self.driver.cudaGetDeviceCount.side_effect = lambda out: 100
        result = self.probe()
        self.assertEqual(result['devices'], [])
        self.assertIn('cudaGetDeviceCount', result['error'])
        self.assertIn('100', result['error'])

    def test_hidden_all_returns_no_visible_devices(self):
        self.devices.clear()
        result = self.probe()
        self.assertEqual(result['devices'], [])
        self.assertIn('visible', result['error'])

    def test_probe_is_cached(self):
        self.probe(); self.probe()
        self.driver.cudaGetDeviceCount.assert_called_once()

    def test_mig_instances_keep_distinct_uuid_and_instance_memory(self):
        self.devices.append(dict(name='MIG slice', memory=2048, cc=(8,0), pci='0000:02:00.0', uuid=bytes(range(1,17))))
        result = self.probe()['devices']
        self.assertEqual(len(result),2)
        self.assertNotEqual(result[0]['uuid'],result[1]['uuid'])
        self.assertEqual(result[1]['memory_mb'],2048)

    def test_unknown_properties_abi_is_rejected(self):
        def version(out): out._obj.value = 13000; return 0
        self.driver.cudaRuntimeGetVersion.side_effect = version
        result = self.probe()
        self.assertEqual(result['devices'], [])
        self.assertIn('Unsupported CUDA runtime version', result['error'])
        self.driver.cudaGetDeviceProperties_v2.assert_not_called()

    def test_missing_runtime_never_uses_cli_indices(self):
        with patch('gpu_hardware._load_cuda_runtime', side_effect=OSError('runtime unavailable')):
            result = self.probe()
        self.assertEqual(result['devices'], [])
        self.assertIn('runtime unavailable', result['error'])

    def test_linux_loader_scans_pip_runtime_library_directory(self):
        from pathlib import Path
        library = Path('runtime-site/nvidia/cuda_runtime/lib/libcudart.so.12')
        with patch('gpu_hardware.sys.platform', 'linux'), patch('gpu_hardware.sys.path', ['runtime-site']), patch('gpu_hardware.Path.is_file', lambda path: path == library), patch('gpu_hardware.ctypes.CDLL', return_value=self.driver) as loader:
            self.assertIs(self.real_loader(), self.driver)
        loader.assert_called_once_with(str(library))

    def test_frozen_loader_scans_bundle_without_changing_path(self):
        import os
        from pathlib import Path
        previous = os.environ.get('PATH')
        library = Path('bundle/nvidia/cuda_runtime/bin/cudart64_12.dll')
        with patch('gpu_hardware.sys._MEIPASS', 'bundle', create=True), patch('gpu_hardware.Path.is_file', lambda path: path == library), patch('gpu_hardware.ctypes.WinDLL', return_value=self.driver, create=True) as loader:
            self.assertIs(self.real_loader(), self.driver)
        loader.assert_called_once_with(str(library))
        self.assertEqual(os.environ.get('PATH'), previous)

    def test_explicit_empty_visibility_never_loads_runtime(self):
        import os
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}):
            result = self.probe()
        self.assertEqual(result['devices'], [])
        self.assertIn('empty', result['error'])
        self.driver.cudaGetDeviceCount.assert_not_called()


if __name__ == '__main__':
    unittest.main()
