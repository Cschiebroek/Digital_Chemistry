Installation
-------------

   ```bash
    mamba env create -f environment.yml
    conda activate chemprop
   ```

Alternatively install from sourece
  ```bash
git clone https://github.com/chemprop/chemprop.git
cd chemprop
conda env create -f environment.yml
conda activate chemprop
  ```

You can download the data used for the models described here from the GitHub OPERA github page https://github.com/kmansouri/OPERA/

TO DO
-------------

- [x] Cleaning up repo for chemprop workflow --> Carl 

- [x] Hyperparameter optimization --> Carl

- [x] Comparison with Opera models (use same splits for STL) --> Enrico
- [x] Varying the number of tasks in the multitask learning --> Enrico main (Riccardo Carl supporting)

- [x] SAMPL challenge benchmark prediction comparison (check if no overlap OPERA-SAMPL) --> Riccardo

- [] Make notebook to make the main plot to use as a template for all the following tasks --> Enrico + Riccardo

- [] Try to do PCA/clustering to select very few datapoints to decide what models will give best performance --> Riccardo at first then Enrico

- [] Assess effect of variance when taking small subset --> Carl

- [] Assess effect of training set size --> Domen

Poster:
https://www.overleaf.com/3869273436fsspfnqhngvr#806db5

- Title: "Deep learning with 20 datapoints"?
- Abstract (inlcuding motivation)
- Methods
- Results
  - External dataset plot
- Conclusions with perspective outlook
  - All the ideas that we don't have time to try

Meeting:
- 23.05.24 1pm - 2pm First sharing of results
  - Comparison with SAMPL: both MTL and STL give very good results, molecules in SAMPL6 are very similar to training data
    - (for the report) get molecules from the training set that are closest to SAMPL6 molecules
    - (for the report) See how well Opera represents a more "general" chemical space by comparing the PCAs
    - (for the report) Find a dataset that IS different? (so maybe MTL will work better than STL)

- 28.05.24 1pm - 5pm Making the poster

Deadline: May 30th 2024 1:45pm (poster session)