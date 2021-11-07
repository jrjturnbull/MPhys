### MPhys Project Repository
#### Folders
- **datafiles**: contains CENTRAL, DATA, SYSTYPE, THEORY files for the relevant experiments
- **dt_comparison**: contains all files pertaining to the validphys data-theory comparisons
- **matrices**: contains all computed matrices from covariance.py (cutoff lines re,pved):
    - *CR*: the correlation matrix
    - *CV*: the covariance matrix
    - *CVN*: the covariance matrix, normalised to the theory
    - *EXP*: the experimental data
    - *NUA*: the nuclear uncertainty array
    - *TH*: the theory_central values
- **output**: contains all output files computed from covariance.py
    - *correlation_matrix*: heatmap of the theory correlation matrix
    - *covariance_matrix*: heatmap of the theory covariance matrix (not normalised)
    - *diagonal_elements*: plot of the covariance matrix diagonal elements, normalised to the experimental data
    - *eigenvalues_data*: raw covariance matrix eigenvalues, normalised to the theory
    - *eigenvalues_plot*: plot of the covariance matrix eigenvalues, normalised to the theory

#### Scripts
- **covariance.py**: the main python script for this project
- **extract_theory.py**: extracts theory_central values from the various computed validphys tables
- **generate_combined.py**: merges data from different experiments (compatible with covariance.py)
- **run_all.sh**: runs the entire project (so far) for the deweighted (non-iterated) data files
