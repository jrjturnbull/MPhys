### MPhys Project Repository
#### Folders
- **datafiles**: contains datafiles for the relevant experiments, including those from validphys (/dtcomparison):
    - *DATA*: the raw data, including all nuclear uncertainties
    - *SYSTYPE*: the column headers for the DATA files
    - *THEORY*: the theoretical predictions, taken from validphys
- **dt_comparison**: contains all files pertaining to the validphys data-theory comparisons
- **ExpCov**: contains validphys files for the experimental covariance matrices
- **matrices**: contains all computed matrices from in pickled format (kinematic cuts applied):
    - *CR*: the theory correlation matrix
    - *CV*: the theory covariance matrix
    - *CVN*: the theory covariance matrix, normalised to the theory
    - *ECR*: the experimental correlation matrix
    - *ECV*: the experimental covariance matrix
    - *EVC*: the theory covariance eigenvectors (for non-zero eigenvalues)
    - *EVCN*: the theory covariance eigenvectors, normalised to the theory
    - *EVL*: the non-zero theory covariance eigenvalues
    - *EVLN*: the theory covariance eigenvalues, normalised to the theory
    - *EXP*: the raw experimental data, taken from datafiles/DATA
    - *NUA*: the nuclear uncertainty array
    - *TH*: the theory_central values, taken from datafiles/THEORY
- **output**: contains all output files computed from output.py
    - *correlation_matrix*: heatmap of the theory correlation matrix
    - *covariance_matrix*: heatmap of the theory covariance matrix (not normalised)
    - *diagonal_elements*: plot of the covariance matrix diagonal elements, normalised to the experimental data
    - *eigenvalues_data*: raw covariance matrix eigenvalues, normalised to the theory
    - *eigenvalues_plot*: plot of the covariance matrix eigenvalues, normalised to the theory

#### Scripts
- **covariance.py**: computes theory correlation/covariance matrices + eigenstuff for the supplied root
- **extract_exp.py**: extracts the experimental covariance/correlation matrices from ExpCov
- **extract_theory.py**: extracts theory_central values from the various computed validphys tables
- **generate_combined.py**: merges data from different experiments (compatible with covariance.py)
- **nuisance.py**: computes the nuisance parameter expectation values for the supplied root (***WORK IN PROGRESS!***)
- **output.py**: generates graphical output files for the supplied root
- **run_dw.sh**: runs the entire project (so far) for the deweighted (non-iterated) data files
- **run_dw_ite.sh**: runs the entire project (so far) for the deweighted iterated data files
