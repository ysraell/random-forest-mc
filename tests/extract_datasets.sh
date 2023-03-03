#!/bin/bash
set -e

mkdir -p /tmp/datasets/

for bziped2 in `ls tests/datasets/*.bz2`;
    do
        csved=`echo $bziped2 | sed -e 's/\.bz2//g' | sed -e 's;tests\/datasets\/;;g'`
        bunzip2 -k -c $bziped2 >/tmp/datasets/${csved}
    done

mkdir -p /tmp/datasets/load_json_empty
echo "" >/tmp/datasets/load_json_empty/empty.json

mkdir -p /tmp/datasets/load_json_csv_like
echo 'a,"!WE@",3' >/tmp/datasets/load_json_csv_like/csv_like.json

mkdir -p /tmp/datasets/load_json_csv_like
echo '{"a": 1}' >/tmp/datasets/load_json_csv_like/csv_like.json

#EOF
