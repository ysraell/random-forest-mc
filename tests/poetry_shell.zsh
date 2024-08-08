#!/usr/bin/env zsh
set -e

if [ -z $1 ] ;
then
    PYTHON_VERSION=`cat .python-version`
else
    PYTHON_VERSION=$1
fi

export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"
pyenv local $PYTHON_VERSION

#poetry shell
