#!/bin/bash
set -e 


rm -fr /tmp/model_dict.json
python -m pytest --durations=0 --cov='./src/' --cov-report term --cov-report=html ./tests/

echo "file://`pwd`/htmlcov/index.html"

#EOF
