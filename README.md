### MPhys Project Repository
#### Folders
- **datafiles**: contains DATA and SYSTYPE files for the relevant experiments
- **dt_comparison**: contains files relating to the validphys data-theory comparisons
- **output**: contains all output files, computed from the python scripts
    - *correlation_matrix*: heatmap of the theory correlation matrix
    - *covariance_matrix*: heatmap of the theory covariance matrix (not normalised)
    - *diagonal_elements*: plot of the covariance matrix diagonal elements, normalised to the experimental data
    - *eigenvalues_data*: raw covariance matrix eigenvalues, normalised to the theory
    - *eigenvalues_plot*: plot of the covariance matrix eigenvalues, normalised to the theory

#### Scripts
- **covariance.py**: the main python script for this project
- **extract_theory.py**: extracts theory_central values from the various computed validphys tables
- **generate_combined.py**: combines data from different experiments into a single DATA + SYSTYPE file pair (compatible with covariance.py)
- **run_all.sh**: runs the entire project (so far) for the deweighted (non-iterated) data files
