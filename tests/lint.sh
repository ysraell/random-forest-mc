#!/usr/bin/env bash
set -e

IGNORE=E501,W503,E203
IGNORE_TEST=E402

/usr/bin/env python -m black .
/usr/bin/env python -m flake8 --ignore ${IGNORE} ./src
/usr/bin/env python -m flake8 --ignore ${IGNORE},${IGNORE_TEST} ./tests

#EOF
