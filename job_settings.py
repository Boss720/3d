"""Versioned complete export settings; validation never mutates a job."""
from __future__ import annotations

import math

from depth_engine import MODEL_SPECS
from stereo import LAYOUTS


SETTINGS_SCHEMA_VERSION = 1
SETTINGS_FIELDS = ('layout', 'strength', 'convergence', 'depth_mode', 'input_size', 'resolution',
                   'quality', 'invert_depth', 'temporal_smoothing', 'edge_refinement', 'execution_mode', 'depth_interval', 'video_decode', 'use_depth_cache')
_ENUMS = {'layout': set(LAYOUTS), 'depth_mode': {'balanced', 'precision'} | set(MODEL_SPECS),
          'resolution': {'source', '1080p'}, 'quality': {'fast', 'balanced', 'high'},
          'execution_mode': {'auto', 'cpu', 'gpu', 'cuda', 'hybrid'}, 'video_decode': {'software', 'amd'}}
_RANGES = {'strength': (0, 10), 'convergence': (0, 1), 'temporal_smoothing': (0, .9)}


def settings_snapshot(job) -> dict:
    return {name: getattr(job, name) if name != 'use_depth_cache' else getattr(job, name, False)
            for name in SETTINGS_FIELDS}


def validate_settings(settings) -> dict:
    # Existing v1 presets/projects/history predate the optional video cadence.
    if type(settings) is dict and 'depth_interval' not in settings:
        settings = {**settings, 'depth_interval': 1}
    if type(settings) is dict and 'video_decode' not in settings:
        settings = {**settings, 'video_decode': 'software'}
    if type(settings) is dict and 'use_depth_cache' not in settings:
        settings = {**settings, 'use_depth_cache': False}
    if type(settings) is not dict or set(settings) != set(SETTINGS_FIELDS):
        raise ValueError('Provide all export settings with no extra fields.')
    if type(settings['depth_interval']) is not int or settings['depth_interval'] not in {1, 2}:
        raise ValueError('Depth interval must be the integer 1 or 2.')
    for name, choices in _ENUMS.items():
        if type(settings[name]) is not str or settings[name] not in choices:
            raise ValueError(f'Choose a supported {name.replace("_", " ")}.')
    if type(settings['input_size']) is not int or settings['input_size'] not in {256, 384, 518}:
        raise ValueError('Model input size must be the integer 256, 384 or 518.')
    for name in ('invert_depth', 'edge_refinement', 'use_depth_cache'):
        if type(settings[name]) is not bool:
            raise ValueError(f'{name.replace("_", " ").capitalize()} must be a boolean.')
    for name, (low, high) in _RANGES.items():
        value = settings[name]
        if type(value) not in {int, float} or not low <= value <= high or not math.isfinite(value):
            raise ValueError(f'{name.replace("_", " ").capitalize()} must be finite and between {low} and {high}.')
    return dict(settings)
