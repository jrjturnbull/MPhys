#!/bin/bash

# RUNS THE ENTIRE PROJECT (SO FAR) FOR THE ITERATED DATA FILES

python3 covariance.py "CHORUSNBPb_dw_ite"
python3 covariance.py "CHORUSNUPb_dw_ite"
python3 covariance.py "DYE605_dw_ite"
python3 covariance.py "NTVNBDMNFe_dw_ite"
python3 covariance.py "NTVNUDMNFe_dw_ite"

python3 generate_combined.py "CHORUSNBPb_dw_ite" "CHORUSNUPb_dw_ite" "DYE605_dw_ite" "NTVNBDMNFe_dw_ite" "NTVNUDMNFe_dw_ite"

python3 covariance.py "CombinedData"