#!/bin/bash

# RUNS THE ENTIRE PROJECT (SO FAR) FOR THE DEWEIGHTED (NON-ITERATED) DATA FILES
# REQUIRES DATA, SYSTYPE, THEORY FILES FOR EACH EXPERIMENT

python3 extract_exp.py
python3 extract_theory.py

python3 covariance.py "CHORUSNBPb_dw"
python3 covariance.py "CHORUSNUPb_dw"
python3 covariance.py "DYE605_dw"
python3 covariance.py "NTVNBDMNFe_dw"
python3 covariance.py "NTVNUDMNFe_dw"

python3 generate_combined.py "CHORUSNBPb_dw" "CHORUSNUPb_dw" "DYE605_dw" "NTVNBDMNFe_dw" "NTVNUDMNFe_dw" "CombinedData_dw"
python3 covariance.py "CombinedData_dw"

python3 nuisance.py "CombinedData_dw"

python3 output.py "CombinedData_dw"
