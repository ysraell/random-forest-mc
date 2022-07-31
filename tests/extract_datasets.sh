#!/bin/bash
set -e 

mkdir -p /tmp/datasets/

for bziped2 in `ls tests/datasets/*.bz2`;
    do
        csved=`echo $bziped2 | sed -e 's/\.bz2//g'`
        bunzip -k -c $bziped2 >/tmp/datasets/${csved}
    done


#EOF