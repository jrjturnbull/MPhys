#!/bin/bash

rm -r covmat/*/output
rm -r chi2/*/*/output

echo
echo "Running nuclear covmat scripts (2 in total)"
cd covmat/nuclear
validphys runcard.yaml
cd ../nuclear_dw
validphys runcard.yaml
echo "Running deuterium covmat scripts (2 in total)"
cd ../deuterium
validphys runcard.yaml
cd ../deuterium_dw
validphys runcard.yaml
echo "Running nNNPDF3.0 covmat scripts (2 in total)"
cd ../30
validphys runcard.yaml
cd ../30_dw
validphys runcard.yaml

cd ../..

echo
echo "Running nuclear chi2 scripts (6 in total)"
cd chi2/nuclear/1_nonuclear
validphys runcard.yaml
cd ../1_nonuclear_t0
validphys runcard.yaml
cd ../2_noshift
validphys runcard.yaml
cd ../2_noshift_t0
validphys runcard.yaml
cd ../3_shift
validphys runcard.yaml
cd ../3_shift_t0
validphys runcard.yaml
echo "Running deuterium chi2 scripts (6 in total)"
cd ../../deuterium/1_nonuclear
validphys runcard.yaml
cd ../1_nonuclear_t0
validphys runcard.yaml
cd ../2_noshift
validphys runcard.yaml
cd ../2_noshift_t0
validphys runcard.yaml
cd ../3_shift
validphys runcard.yaml
cd ../3_shift_t0
validphys runcard.yaml
echo "Running nNNPDF3.0 chi2 scripts (6 in total)"
cd ../../30/1_nonuclear
validphys runcard.yaml
cd ../1_nonuclear_t0
validphys runcard.yaml
cd ../2_noshift
validphys runcard.yaml
cd ../2_noshift_t0
validphys runcard.yaml
cd ../3_shift
validphys runcard.yaml
cd ../3_shift_t0
validphys runcard.yaml
cd ../../..