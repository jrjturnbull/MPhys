#!/bin/bash

# RUNS THE ENTIRE PROJECT (SO FAR) FOR THE DEWEIGHTED ITERATED DATA FILES
# REQUIRES DATA, SYSTYPE, THEORY FILES FOR EACH EXPERIMENT

python3 covariance.py "CHORUSNBPb_dw_ite"
python3 covariance.py "CHORUSNUPb_dw_ite"
python3 covariance.py "DYE605_dw_ite"
python3 covariance.py "NTVNBDMNFe_dw_ite"
python3 covariance.py "NTVNUDMNFe_dw_ite"

python3 generate_combined.py "CHORUSNBPb_dw_ite" "CHORUSNUPb_dw_ite" "DYE605_dw_ite" "NTVNBDMNFe_dw_ite" "NTVNUDMNFe_dw_ite" "CombinedData_dw_ite"

python3 covariance.py "CombinedData_dw_ite"
python3 output.py "CombinedData_dw_ite"
