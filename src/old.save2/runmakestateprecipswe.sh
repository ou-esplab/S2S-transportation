#!/bin/bash
set -xve

. /home/$USER/miniconda3/etc/profile.d/conda.sh
conda activate s2stransportation

./MakePrecipStates.py

