#!/bin/bash

set -xe

WORKDIR=$(cd $(dirname $0);pwd)

cd $WORKDIR/$1

poetry install
sh -c 'poetry run pytest -vv . ; ret=$?; [ $ret = 5 ] && exit 0 || exit $ret'