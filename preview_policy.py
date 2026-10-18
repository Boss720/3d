"""Source-detail limits for still previews, separate from neural input policy.

Decoded dimensions are checked after OpenCV decode; these limits do not promise
a process-memory peak or bound the decoder's initial allocation.
"""
from __future__ import annotations

from dataclasses import dataclass


PREVIEW_UPLOAD_BYTES = 24 * 1024 * 1024
FULL_MAX_PIXELS = 12_000_000
FULL_MAX_DIMENSION = 8192
PREVIEW_POLICY_VERSION = 'preview-source-v1'


class PreviewPolicyError(ValueError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def validate_preview_detail(detail: str) -> None:
    if detail not in {'proxy', 'full'}:
        raise PreviewPolicyError('Choose proxy or full preview detail. Use proxy for a smaller preview.')


def validate_preview_upload_size(size: int) -> None:
    if size > PREVIEW_UPLOAD_BYTES:
        raise PreviewPolicyError('The preview frame exceeds 24 MiB. Use proxy preview or a smaller image.', 413)


@dataclass(frozen=True)
class PreviewSourcePolicy:
    original_width: int
    original_height: int
    width: int
    height: int
    detail: str

    @property
    def is_proxy(self) -> bool:
        return self.detail == 'proxy'

    @property
    def cache_identity(self) -> tuple:
        return PREVIEW_POLICY_VERSION, self.detail, self.width, self.height


def preview_source_policy(width: int, height: int, detail: str = 'proxy') -> PreviewSourcePolicy:
    validate_preview_detail(detail)
    if width < 1 or height < 1:
        raise PreviewPolicyError('Preview dimensions must be positive. Try another image or proxy preview.')
    if detail == 'full':
        if width * height > FULL_MAX_PIXELS or max(width, height) > FULL_MAX_DIMENSION:
            raise PreviewPolicyError(
                'Full-detail preview supports at most 12,000,000 decoded pixels and 8192 pixels per dimension. '
                'Use proxy preview or resize the image.', 413)
        out_width, out_height = width, height
    else:
        scale = min(1.0, 960 / width, 540 / height)
        out_width, out_height = max(1, round(width * scale)), max(1, round(height * scale))
    return PreviewSourcePolicy(width, height, out_width, out_height, detail)
