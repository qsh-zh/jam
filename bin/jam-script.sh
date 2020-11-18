#!/bin/bash -E

JAMROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/../ && pwd)"
SCRIPT=$(basename ${BASH_SOURCE[0]})
SCRIPT=${SCRIPT:4}

source $JAMROOT/bin/_jam-init.sh $JAMROOT

exec python3 "$JAMROOT/scripts/$SCRIPT.py" "$@"
