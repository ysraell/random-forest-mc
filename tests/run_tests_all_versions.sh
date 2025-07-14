#!/bin/bash
set -e 

# run install_build_dependencies.sh

for version in `cat tests/versions.txt |grep -v '#'`;
do
    pyenv install -s $version
    pyenv local $version
    pip install -U -r tests/requirements.txt
    pip uninstall -y random-forest-mc
    bash -x tests/run_tests.sh || echo "Something wrong with $version"
done

exit 0

# with JIT:
# use https://github.com/AdrianDAlessandro/pyenv-suffix
for version in `cat tests/versions-jit-nogil.txt |grep -v '#'`;
do
    export PYENV_VERSION_SUFFIX="-jit" 
    export PYTHON_CONFIGURE_OPTS='--enable-experimental-jit'
    pyenv install -f $version
    pyenv local $version$PYENV_VERSION_SUFFIX
    pip install -U -r tests/requirements.txt
    pip uninstall -y random-forest-mc
    bash -x tests/run_tests.sh || echo "Something wrong with $version-$PYENV_VERSION_SUFFIX"

    export PYTHON_GIL=0

    export PYENV_VERSION_SUFFIX="-nogil" 
    export PYTHON_CONFIGURE_OPTS='--disable-gil'
    pyenv install -f $version
    pyenv local $version$PYENV_VERSION_SUFFIX
    pip install -U -r tests/requirements.txt
    pip uninstall -y random-forest-mc
    bash -x tests/run_tests.sh || echo "Something wrong with $version-$PYENV_VERSION_SUFFIX"

    export PYENV_VERSION_SUFFIX="-nogil-jit" 
    export PYTHON_CONFIGURE_OPTS='--disable-gil --enable-experimental-jit'
    pyenv install -f $version
    pyenv local $version$PYENV_VERSION_SUFFIX
    pip install -U -r tests/requirements.txt
    pip uninstall -y random-forest-mc
    bash -x tests/run_tests.sh || echo "Something wrong with $version-$PYENV_VERSION_SUFFIX"
done