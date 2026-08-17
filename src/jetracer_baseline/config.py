# -*- coding: utf-8 -*-
"""Nap config YAML.

Python 3.6 compatible (Jetson Nano / JetPack 4.6). Khong dung dataclasses,
khong dung f-string, khong dung type union `X | Y`.
"""

import copy
import io
import os

import yaml


class Config(object):
    """Truy cap config kieu `cfg.get('control.v_max', 0.2)` hoac `cfg['control']`."""

    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def get(self, dotted_key, default=None):
        node = self._data
        for part in dotted_key.split('.'):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, dotted_key, value):
        parts = dotted_key.split('.')
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def as_dict(self):
        return copy.deepcopy(self._data)


def _deep_merge(base, override):
    """Merge `override` vao `base` theo chieu sau, tra ve dict moi."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path, overrides=None):
    """Nap `path`, roi ap lan luot cac file trong `overrides` (list duong dan)."""
    if not os.path.exists(path):
        raise IOError('Khong tim thay config: ' + path)
    # encoding tuong minh: config co comment tieng Viet, Windows mac dinh cp1252
    with io.open(path, 'r', encoding='utf-8') as fh:
        data = yaml.safe_load(fh) or {}

    for extra in (overrides or []):
        if not os.path.exists(extra):
            raise IOError('Khong tim thay override config: ' + extra)
        with io.open(extra, 'r', encoding='utf-8') as fh:
            data = _deep_merge(data, yaml.safe_load(fh) or {})

    return Config(data)
