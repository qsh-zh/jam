#!/bin/bash

JAMROOT=$1

export PYTHONPATH=$JAMROOT:./:$PYTHONPATH
fname=`JAM_IMPORT_ALL=FALSE python3 $JAMROOT/bin/_gen-rc.py`
source $fname
rm -f $fname
