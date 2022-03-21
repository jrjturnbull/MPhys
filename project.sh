#!/bin/bash

sh validphys.sh

echo
echo "Running run.sh for nuclear data"
sh run.sh nuclear
echo
echo "Running run.sh for deuterium data"
sh run.sh deuterium
echo
echo "Running run.sh for nNNPDF3.0 data"
sh run.sh 30