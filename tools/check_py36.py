# -*- coding: utf-8 -*-
"""Chan cu phap chi co tu Python 3.7+ lot vao repo.

    python tools/check_py36.py

Vi sao can: Jetson Nano (JetPack 4.6 / Ubuntu 18.04) chay Python 3.6. Code viet
tren laptop Python 3.10+ chay ngon lanh o nha roi chet ngay dong import khi mang
len xe - va luc do dang dung gio chay chung 5 xe / 10 doi. Chay check nay TRUOC
moi lan dong bo code len xe.
"""

from __future__ import print_function

import ast
import io
import os
import sys

SKIP_DIRS = ('.git', '__pycache__', 'logs', 'reports', 'data', '.venv', 'venv')
PY37_MODULES = ('dataclasses', 'zoneinfo', 'graphlib', 'contextvars')


def check_file(path):
    problems = []
    with io.open(path, encoding='utf-8') as fh:
        src = fh.read()
    try:
        tree = ast.parse(src, path)
    except SyntaxError as exc:
        return [(exc.lineno or 0, 'khong parse duoc: %s' % exc.msg)]

    for node in ast.walk(tree):
        if node.__class__.__name__ == 'NamedExpr':
            problems.append((node.lineno, 'toan tu walrus := (can Python 3.8+)'))
        elif isinstance(node, ast.JoinedStr):
            problems.append((node.lineno, 'f-string (dung .format() cho dong bo)'))
        elif isinstance(node, ast.AsyncFunctionDef):
            problems.append((node.lineno, 'async def'))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            names += [a.name for a in node.names]
            for name in names:
                if name and name.split('.')[0] in PY37_MODULES:
                    problems.append((node.lineno,
                                     'import %s (can Python 3.7+)' % name))
    return problems


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    total = 0
    failed = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if not name.endswith('.py'):
                continue
            path = os.path.join(dirpath, name)
            total += 1
            for lineno, why in check_file(path):
                failed += 1
                print('%s:%d  %s' % (os.path.relpath(path, root), lineno, why))

    if failed:
        print('')
        print('%d van de - code nay se KHONG chay tren Jetson Nano.' % failed)
        return 1
    print('OK - %d file .py tuong thich cu phap Python 3.6' % total)
    return 0


if __name__ == '__main__':
    sys.exit(main())
