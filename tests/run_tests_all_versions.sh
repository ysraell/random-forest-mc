#!/bin/bash
set -e 

# run install_build_dependencies.sh

for version in `cat tests/versions.txt`;
do
    pyenv install -s $version
    pyenv local $version
    pip install -U -r tests/requirements.txt
    pip install .
    bash -x tests/run_tests.sh
done