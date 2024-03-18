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

May 6th 2024
- Which properties? And how to combine? (Together?)
- Chemical space analysis (Carl, Riccardo) --> generalize to new molecules? 
- HP optimization (Together?, Carl)
- Compare MTL vs STL (Enrico + Riccardo)

May 30th 2024 (poster session)
- Make poster

Unspecified date
- Model architectures (AttentiveFP vs ChemProp) (low priority)
- Analyze attentions (Enrico)
- Report results for different train/test splits
