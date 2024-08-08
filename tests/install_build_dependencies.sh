#!/bin/bash
set -e 

apt-get update
apt-get install -y build-essential \
    libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget \
    curl llvm gettext libncurses5-dev \
    tk-dev tcl-dev blt-dev libgdbm-dev \
    git python3-dev aria2 lzma liblzma-dev

#EOF