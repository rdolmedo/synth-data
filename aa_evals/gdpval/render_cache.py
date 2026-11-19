"""Integrity of evaluator-produced PNG cache files (not submitted-file validity).

This module also runs as a sibling of /code/render.py inside the existing SIF.
Pillow is imported only while checking an emitted PNG, not by the host judge.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


class RenderIntegrityError(RuntimeError):
    """The evaluator could not safely publish/read its generated PNG cache."""

    def __init__(self, path, detail):
        self.path = str(path)
        self.detail = str(detail)
        super().__init__(f'render PNG cache integrity at {self.path}: {self.detail}')

    def as_dict(self):
        return {'schema_version': 1, 'kind': 'gdpval_render_png_cache_integrity',
                'path': self.path, 'detail': self.detail}


def publish_png(destination, writer):
    """Keep the existing encoder/options; publish only a fully decoded nonempty PNG."""
    from PIL import Image

    destination = Path(destination)
    temporary = None
    try:
        fd, name = tempfile.mkstemp(prefix='.' + destination.stem + '-', suffix='.png', dir=destination.parent)
        os.close(fd)
        temporary = Path(name)
        writer(temporary)
        if temporary.stat().st_size == 0:
            raise ValueError('generated PNG is empty')
        with Image.open(temporary) as image:
            image.load()
            if image.format != 'PNG' or image.width <= 0 or image.height <= 0:
                raise ValueError('generated image is not a valid nonempty PNG')
        os.replace(temporary, destination)
    except Exception as error:
        raise RenderIntegrityError(destination, f'{type(error).__name__}: {error}') from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # keep the original typed write/decode error; no partial file is advertised
