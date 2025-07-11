#!/bin/bash
set -e

TMP_DIR=$1

if [ -z "$TMP_DIR" ]; then
    echo "Usage: $0 <tmp_dir>"
    exit 1
fi

DATASETS_DIR=$TMP_DIR/datasets/
mkdir -p $DATASETS_DIR

for bziped2 in `ls tests/datasets/*.bz2`;
    do
        csved=`echo $bziped2 | sed -e 's/\.bz2//g' | sed -e 's;tests\/datasets\/;;g'`
        bunzip2 -k -c $bziped2 >$DATASETS_DIR/${csved}
    done

mkdir -p $DATASETS_DIR/load_json_empty
echo "" >$DATASETS_DIR/load_json_empty/empty.json

mkdir -p $DATASETS_DIR/load_json_csv_like
echo 'a,"!WE@",3' >$DATASETS_DIR/load_json_csv_like/csv_like.json

mkdir -p $DATASETS_DIR/load_json_keyword
echo '{"a": 1}' >$DATASETS_DIR/load_json_keyword/True.json

#EOF
