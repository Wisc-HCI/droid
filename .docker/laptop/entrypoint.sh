#!/bin/bash

# activate conda
# NOTE: use conda.sh, not `source ~/miniconda3/bin/activate`. The latter inherits
# this script's positional args ($@ = the user command) and feeds them to the
# legacy activate script, causing "activate does not accept more than one argument".
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robot

# run user command
exec "$@"
