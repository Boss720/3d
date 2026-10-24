"""Budgeted complete-export measurements through the ordinary export adapter.

This module never creates files, starts processes, or changes user settings.
The caller must reserve the inference/queue slot for the entire measurement.
The adapter owns per-run paths and processes, finalizes and validates each
output before returning, and cleans partial files on failure or cancellation.
Successful outputs remain owned by the caller for A/B review and expiry.
"""
from __future__ import annotations

import copy
import math
import statistics
import threading
import time

from calibration import fingerprint


SCHEMA_VERSION = 1
REPEATS = 2


def _number(value, name, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f'{name} must be a finite number from {minimum} to {maximum}.')
    return value


def export_candidates(baseline_settings: dict) -> list[dict]:
    """Keep the selected model and media settings; vary at most one control."""
    baseline = copy.deepcopy(baseline_settings)
    if baseline.get('input_size') not in (256, 384, 518):
        raise ValueError('Choose input detail 256, 384, or 518 for calibration.')
    if baseline.get('quality') not in ('fast', 'balanced', 'high'):
        raise ValueError('Choose fast, balanced, or high export quality.')
    if not isinstance(baseline.get('depth_mode'), str) or not baseline['depth_mode']:
        raise ValueError('Choose a depth model for calibration.')
    baseline['use_depth_cache'] = False
    rows = []
    seen = set()
    for candidate_id, settings in (
        ('baseline', baseline),
        ('input256', {**baseline, 'input_size': 256}),
        ('input518', {**baseline, 'input_size': 518}),
        ('encodingfast', {**baseline, 'quality': 'fast'}),
    ):
        key = (settings['input_size'], settings['quality'])
        if key in seen:
            continue
        seen.add(key)
        rows.append({'candidate_id': candidate_id, 'settings': copy.deepcopy(settings),
                     'samples_seconds': [], 'runs': [], 'outputs': [],
                     'median_seconds': None, 'range_seconds': None, 'error': None})
    return rows


def measure_exports(run_sample, identity: dict, baseline_settings: dict,
                    budget_seconds: float = 120, cancel_event=None,
                    clock=time.monotonic, on_progress=None, started_at=None) -> dict:
    """Measure two full exports for each of at most four exact candidates.

    ``run_sample(settings, sample, cancel_event)`` receives isolated dict copies.
    ``sample`` repeats identity['sample'] with candidate_id and repeat_index
    (0 or 1). It must return completed=True, validated=True, a nonempty output
    dict suitable for A/B review, and execution metadata with active_provider.
    Optional context, timings, encoder, and validation details are retained.
    Timing wraps the entire callback, including finalization and validation.
    No stage-time sum is used. An in-progress sample may exceed the budget;
    the adapter must handle cooperative cancellation of its normal pipeline.

    ``on_progress(snapshot)`` receives candidate_id, repeat_index, attempts,
    total_samples, elapsed_seconds, and phase ('starting' or 'finished').
    Caller setup may be counted via a monotonic ``started_at`` from the same
    clock. There is no background work or automatic settings application.
    """
    _number(budget_seconds, 'Calibration budget seconds', 1, 120)
    identity = copy.deepcopy(identity)
    selected_settings = copy.deepcopy(baseline_settings)
    sample = identity.get('sample')
    if not isinstance(sample, dict):
        raise ValueError('Calibration identity must contain a sample interval.')
    _number(sample.get('start_seconds'), 'Sample start seconds', 0, 10**9)
    _number(sample.get('duration_seconds'), 'Sample duration seconds', .05, 5)
    candidates = export_candidates(selected_settings)
    if len(candidates) < 2:
        raise ValueError('Calibration needs at least two distinct candidates.')
    report_fingerprint = fingerprint({'identity': identity, 'settings': selected_settings,
                                      'schema_version': SCHEMA_VERSION})
    cancel_event = cancel_event if cancel_event is not None else threading.Event()
    started = clock() if started_at is None else started_at
    attempts = 0
    provider_identity = None
    stop = False

    def progress(phase, row, repeat_index):
        if on_progress is not None:
            on_progress({'phase': phase, 'candidate_id': row['candidate_id'],
                         'repeat_index': repeat_index, 'attempts': attempts,
                         'total_samples': len(candidates) * REPEATS,
                         'elapsed_seconds': max(0.0, clock() - started)})

    for row in candidates:
        for repeat_index in range(REPEATS):
            if cancel_event.is_set() or clock() - started >= budget_seconds:
                stop = True
                break
            progress('starting', row, repeat_index)
            # Progress adapters may perform work or receive cancellation.
            if cancel_event.is_set() or clock() - started >= budget_seconds:
                stop = True
                break
            request = {**copy.deepcopy(sample), 'candidate_id': row['candidate_id'],
                       'repeat_index': repeat_index}
            attempts += 1
            before = clock()
            run = {'repeat_index': repeat_index, 'sample': copy.deepcopy(request),
                   'timing_context': 'first_export' if repeat_index == 0 else 'repeated_export',
                   'completed': False, 'validated': False, 'error': None}
            try:
                result = run_sample(copy.deepcopy(row['settings']), request, cancel_event)
                elapsed = max(0.0, clock() - before)
                run['wall_seconds'] = elapsed
                if not isinstance(result, dict):
                    raise RuntimeError('Export adapter did not return output evidence.')
                run['result'] = copy.deepcopy(result)
                if result.get('completed') is not True or result.get('validated') is not True:
                    raise RuntimeError('Export did not report completed and validated output.')
                output = result.get('output')
                if not isinstance(output, dict) or not output:
                    raise RuntimeError('Export did not retain an output for A/B review.')
                execution = result.get('execution')
                if not isinstance(execution, dict) or not execution.get('active_provider'):
                    raise RuntimeError('Export did not report its actual execution provider.')
                if execution.get('fallback_reason') or execution.get('inference_succeeded') is False:
                    raise RuntimeError('Execution fell back or inference failed; retry explicitly on the actual device.')
                actual = tuple(execution.get(key) for key in ('active_provider', 'device_id', 'device'))
                if provider_identity is None:
                    provider_identity = actual
                elif actual != provider_identity:
                    raise RuntimeError('Actual provider or device changed; exports are not comparable.')
                if elapsed <= 0 or not math.isfinite(elapsed):
                    raise RuntimeError('Timing resolution was insufficient for this full export.')
                run.update(completed=True, validated=True)
                row['samples_seconds'].append(elapsed)
                row['outputs'].append(copy.deepcopy(output))
            except Exception as exc:
                run['wall_seconds'] = max(0.0, clock() - before)
                run['error'] = str(exc)[:512] or type(exc).__name__
                row['error'] = run['error']
                # The callback owns cleanup. A failure may mean cleanup failed,
                # so starting more inference cannot safely establish evidence.
                stop = True
            row['runs'].append(run)
            progress('finished', row, repeat_index)
            if stop or cancel_event.is_set():
                stop = True
                break
        if stop:
            break

    for row in candidates:
        samples = row['samples_seconds']
        row['median_seconds'] = statistics.median(samples) if len(samples) == REPEATS and not row['error'] else None
        row['range_seconds'] = [min(samples), max(samples)] if samples else None
    canceled = cancel_event.is_set()
    sufficient = not canceled and all(row['median_seconds'] is not None for row in candidates)
    recommendation = None
    if sufficient:
        fastest = min(candidates, key=lambda row: row['median_seconds'])
        recommendation = {'candidate_id': fastest['candidate_id'],
                          'settings': copy.deepcopy(fastest['settings']),
                          'median_seconds': fastest['median_seconds'],
                          'speedup_over_baseline': candidates[0]['median_seconds'] / fastest['median_seconds'],
                          'basis': 'Lowest median wall time among the measured candidates for this sample only.',
                          'quality_verified': False}
    elapsed = max(0.0, clock() - started)
    return {'schema_version': SCHEMA_VERSION, 'fingerprint': report_fingerprint,
            'identity': identity, 'selected_settings': selected_settings,
            'effective_baseline_settings': copy.deepcopy(candidates[0]['settings']),
            'budget_seconds': budget_seconds, 'elapsed_seconds': elapsed,
            'budget_exceeded': elapsed > budget_seconds, 'attempts': attempts,
            'state': 'canceled' if canceled else 'completed' if sufficient else 'incomplete',
            'sufficient_evidence': sufficient, 'settings_applied': False,
            'fastest_measured_candidate_id': recommendation['candidate_id'] if recommendation else None,
            'recommendation': recommendation, 'candidates': candidates,
            'scope': 'Complete exports of the same bounded sample interval, including encoder finalization and output validation. '
                     'If this sample is a proxy, these are proxy export timings; they do not predict full source export speed.',
            'limitations': [
                'The budget is checked between complete exports. An in-progress inference or encoder finalization may exceed it.',
                'Two exports per candidate are retained. First exports may include cold session setup; repeats may reuse sessions. '
                'No separate warmup is excluded and ordering may affect results.',
                'Depth cache is disabled for every candidate, including the baseline, to measure conversion work consistently.',
                'Lower input detail may lose fine edges. Fast encoding may reduce output quality or increase output size.',
                'The fastest measured settings are a speed suggestion only. Inspect A/B outputs for quality before applying.',
                'The selected depth model, sample interval, and other export settings stay fixed. No settings were applied.',
                'Source content, proxy dimensions, runtime, hardware, and encoder affect these results; no full-video estimate is made.',
            ]}
