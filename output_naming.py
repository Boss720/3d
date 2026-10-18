"""Bounded output filename templates, separate from export settings."""
from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUTPUT_NAME_TEMPLATE = '{name}_{layout}'
NAMING_SCHEMA_VERSION = 1
MAX_TEMPLATE_LENGTH = 120
NAMING_FIELDS = ('name', 'layout', 'model', 'date')
_TOKEN = re.compile(r'\{([^{}]*)\}')
_INVALID_LITERAL = frozenset('<>:"/\\|?*{}')
_WINDOWS_RESERVED = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{prefix}{number}' for prefix in ('COM', 'LPT') for number in range(1, 10)}


def _parse_template(template: str) -> list[tuple[bool, str]]:
    if type(template) is not str or not template.strip() or len(template) > MAX_TEMPLATE_LENGTH:
        raise ValueError('Output naming template must be a nonempty string of at most 120 characters.')
    parts = []
    previous = 0
    for match in _TOKEN.finditer(template):
        literal = template[previous:match.start()]
        _validate_literal(literal)
        if match[1] not in NAMING_FIELDS:
            raise ValueError('Use only {name}, {layout}, {model} and {date}; formatting and field lookups are unsupported.')
        parts.extend(((False, literal), (True, match[1])))
        previous = match.end()
    literal = template[previous:]
    _validate_literal(literal)
    parts.append((False, literal))
    return parts


def _validate_literal(literal: str) -> None:
    if any(character in _INVALID_LITERAL or unicodedata.category(character) in {'Cc', 'Cf', 'Cs'} for character in literal):
        raise ValueError('Output naming template contains unmatched braces, path separators, control characters or invalid filename characters.')


def validate_output_template(template: str) -> str:
    _parse_template(template)
    return template


def unique_job_identity(job_id: str) -> str:
    try:
        return uuid.UUID(job_id).hex
    except (ValueError, TypeError, AttributeError):
        return hashlib.sha256(str(job_id).encode('utf-8')).hexdigest()[:32]


def render_output_name(template: str, filename: str, layout: str, model: str,
                       created_at: float, job_id: str, media_type: str) -> str:
    parts = _parse_template(template)
    if media_type not in {'image', 'video'}:
        raise ValueError('Output naming requires an image or video job.')
    values = {'name': Path(filename.replace('\\', '/')).stem,
              'layout': layout.lower(), 'model': model,
              'date': datetime.fromtimestamp(created_at, timezone.utc).strftime('%Y%m%d')}
    # Substitute parsed tokens once. Source braces are data, never another
    # formatting pass, attribute lookup or Python format expression.
    rendered = ''.join(values[value] if is_token else value for is_token, value in parts)
    stem = re.sub(r'[^A-Za-z0-9._ -]+', '_', rendered).strip(' ._')[:100].rstrip(' ._')
    stem = stem or 'stereo-output'
    # Windows recognizes device names before the first dot, even when another
    # extension/UUID follows. A prefix keeps these rendered names ordinary files.
    if '.' in stem and stem.split('.', 1)[0].rstrip(' ').upper() in _WINDOWS_RESERVED:
        stem = '_' + stem[:99]
    suffix = 'png' if media_type == 'image' else 'mp4'
    return f'{stem}_{unique_job_identity(job_id)}.{suffix}'
