Installation
-------------

   ```bash
    mamba env create -f environment.yml
    conda activate mtl
   ```

This project also currently requires Marc Lehner's dash charges. Currently (05.03.2024), the public Github version is not working properly so you need to ask him for access to his private one.

Once you cloned the dash-tree repo, just go there and do

   ```bash
    python setup.py install
   ```

The current environment should hopefully do the job.

If you want to use dash trees, their path should be added in "utils_data.py"

TO DO
-------------

- Cleaning up repo for chemprop workflow --> Carl

- Hyperparameter optimization --> Carl

- Comparison with Opera models (use same splits for STL) --> Enrico
- Varying the number of tasks in the multitask learning --> Enrico main (Riccardo Carl supporting)

- SAMPL challenge benchmark prediction comparison (check if no overlap OPERA-SAMPL) --> Riccardo

- Scaffold split OPERA data --> Domen (Carl supporting)
   - https://jcheminf.biomedcentral.com/articles/10.1186/s13321-023-00787-9

Poster:
https://www.overleaf.com/3869273436fsspfnqhngvr#806db5

- Abstract (inlcuding motivation)
- Methods
- Results
- Conclusions

Meeting:
- 23.05.24 1pm - 2pm First sharing of results
- 28.05.24 1pm - 5pm Making the poster

Deadline: May 30th 2024 3:30pm (poster session)