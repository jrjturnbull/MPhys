### MPhys Project Repository
#### Folders
- **datafiles**: contains datafiles for the relevant experiments, including those from validphys (no cuts applied):
    - *DATA*: the raw data, including all nuclear uncertainties
    - *SYSTYPE*: the column headers for the DATA files
    - *THEORY*: the theoretical predictions, taken from validphys
- **dt_comparison**: contains all files pertaining to the validphys data-theory comparisons
- **dt_comparison_internal**: as above but with internal cuts applied
- **ExpCov**: contains validphys files for the experimental covariance matrices
- **matrices**: contains all computed matrices from in pickled format (kinematic cuts applied):
    - *CR*: the theory correlation matrix
    - *CV*: the theory covariance matrix
    - *CVN*: the theory covariance matrix, normalised to the theory
    - *ECR*: the experimental correlation matrix
    - *ECV*: the experimental covariance matrix
    - *EVC*: the theory covariance eigenvectors (for non-zero eigenvalues)
    - *EVL*: the non-zero theory covariance eigenvalues
    - *EXP*: the raw experimental data, taken from datafiles/DATA
    - *NPE*: the nuisance parameter expectation values
    - *NUA*: the nuclear uncertainty array
    - *TH*: the theory_central values, taken from datafiles/THEORY
    - *XCR*: the pdf correlation 'X' matrix
    - *XCV*: the pdf covariance 'X' matrix
- **output**: contains all output files computed from output.py

#### Scripts
- **autoprediction.py**: computes the autoprediction shifts and matrices ***WIP!***
- **covariance.py**: computes theory correlation/covariance matrices + eigenstuff for the supplied root
- **extract_exp.py**: extracts the experimental covariance/correlation matrices from ExpCov
- **extract_theory.py**: extracts theory_central values from the various computed validphys tables
- **generate_combined.py**: merges data from different experiments (compatible with covariance.py)
- **nuisance.py**: computes the nuisance parameter expectation values for the supplied root ***WIP!***
- **output.py**: generates graphical output files for the supplied root ***WIP!***
- **pdf_covariance.py**: computes the pdf covariance/correlation X matrices
- **run_dw.sh**: runs the entire project (so far) for the deweighted (non-iterated) data files
- **run_dw_ite.sh**: runs the entire project (so far) for the deweighted iterated data files (unsupported)
