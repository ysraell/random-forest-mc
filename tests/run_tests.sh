#!/bin/bash

rm -fr /tmp/model_dict.json
poetry run pytest --durations=0 --cov='./src/' --cov-fail-under=1 --cov-report term --cov-report=html ./tests/

echo "file://`pwd`/htmlcov/index.html"

#EOF
