"""Opt-in, budgeted depth-input measurements; never applies settings itself."""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import threading
import time
import uuid
from pathlib import Path


SCHEMA_VERSION = 1


def fingerprint(identity: dict) -> str:
    return hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()


def measure_inputs(predict, identity: dict, budget_seconds: float = 15,
                   clock=time.monotonic, sizes=(256, 384, 518), started_at=None) -> dict:
    """One warmup per input shape, then two measured calls per candidate.

    The budget includes warmup and failed work. An inference already in progress
    cannot be interrupted; report any overrun. Results describe this one frame.
    """
    if isinstance(budget_seconds, bool) or not math.isfinite(budget_seconds) or not 1 <= budget_seconds <= 30:
        raise ValueError('Choose a calibration time budget from 1 to 30 seconds.')
    started = clock() if started_at is None else started_at
    candidates = {size: {'input_size': size, 'samples_seconds': [], 'warmup_seconds': [],
                         'execution': None, 'error': None} for size in sizes}
    attempts = 0
    provider_identity = None
    for round_index in range(3):
        for size in sizes:
            candidate = candidates[size]
            if candidate['error'] or len(candidate['samples_seconds']) >= 2:
                continue
            if clock() - started >= budget_seconds:
                break
            before = clock()
            attempts += 1
            try:
                execution = predict(size)
                elapsed = max(0.0, clock() - before)
                candidate['execution'] = execution
                if execution.get('inference_succeeded') is not True:
                    raise RuntimeError('Inference did not report success.')
                if not execution.get('active_provider') or not isinstance(execution.get('session_reused'), bool):
                    raise RuntimeError('Runtime did not report provider/session metadata. Upgrade the backend and retry.')
                # Fallback makes requested GPU measurements incomparable.
                if execution.get('fallback_reason'):
                    raise RuntimeError('The requested execution fell back; calibrate CPU explicitly or retry GPU.')
                actual = tuple(execution.get(key) for key in ('active_provider', 'device_id', 'device'))
                if provider_identity is None:
                    provider_identity = actual
                elif actual != provider_identity:
                    raise RuntimeError('Actual provider or device changed during calibration; results are not comparable. Retry.')
                previous = candidate.get('actual_dimensions')
                dimensions = [execution.get('input_width'), execution.get('input_height')]
                if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in dimensions):
                    raise RuntimeError('Actual model input dimensions were not reported. Upgrade the backend and retry.')
                if elapsed <= 0:
                    raise RuntimeError('Timing resolution was insufficient for this sample. Retry.')
                if previous is not None and dimensions != previous:
                    raise RuntimeError('Actual model dimensions changed between repeated samples. Retry.')
                candidate['actual_dimensions'] = dimensions
                field = 'warmup_seconds' if not candidate['warmup_seconds'] or execution.get('session_reused') is False else 'samples_seconds'
                candidate[field].append(elapsed)
            except Exception as exc:
                candidate['error'] = str(exc)[:320]
        if clock() - started >= budget_seconds:
            break
    rows = list(candidates.values())
    for row in rows:
        samples = row['samples_seconds']
        row['median_seconds'] = statistics.median(samples) if len(samples) >= 2 else None
        row['range_seconds'] = [min(samples), max(samples)] if samples else None
    complete = all(row['median_seconds'] is not None and not row['error'] for row in rows)
    elapsed = max(0.0, clock() - started)
    winner = min(rows, key=lambda row: row['median_seconds'])['input_size'] if complete else None
    return {'schema_version': SCHEMA_VERSION, 'fingerprint': fingerprint(identity),
            'identity': identity, 'budget_seconds': budget_seconds, 'elapsed_seconds': elapsed,
            'budget_exceeded': elapsed > budget_seconds, 'attempts': attempts,
            'sufficient_evidence': complete, 'fastest_measured_input_size': winner,
            'candidates': rows,
            'scope': 'Depth-input inference on one proxy frame, including preprocessing/postprocessing. No export or encoder comparison.',
            'limitations': ['Lower input detail may lose fine edges. No settings were changed.',
                            'Each input shape has a warmup; session initialization and warmup count against the time budget.',
                            'One in-progress inference can exceed the budget; it cannot be interrupted.',
                            'This is source-specific timing evidence, not a universal quality or speed ranking.']}


class CalibrationStore:
    """Atomic, bounded opt-in measurement history. Corrupt state is not trusted."""
    def __init__(self, path: Path, max_entries: int = 16):
        if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError('max_entries must be a positive integer.')
        self.path = path
        self.max_entries = max_entries
        self.lock = threading.Lock()

    def save(self, report: dict):
        with self.lock:
            try:
                def reject_constant(value):
                    raise ValueError('Non-finite JSON constant: ' + value)
                stored = json.loads(self.path.read_text(encoding='utf-8'), parse_constant=reject_constant)
                entries = stored['entries'] if stored.get('schema_version') == SCHEMA_VERSION else []
                if not isinstance(entries, list):
                    entries = []
            except (OSError, ValueError, TypeError, KeyError, AttributeError):
                entries = []
            entries = [entry for entry in entries if isinstance(entry, dict) and entry.get('fingerprint') != report['fingerprint']]
            entries.append(report)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.tmp')
            try:
                temporary.write_text(json.dumps({'schema_version': SCHEMA_VERSION, 'entries': entries[-self.max_entries:]},
                                                allow_nan=False, indent=2), encoding='utf-8')
                os.replace(temporary, self.path)
            finally:
                temporary.unlink(missing_ok=True)
