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
- Review and complete the Project Proposal Template: https://docs.google.com/document/d/1ZiWIgKuQqnhzkXxxvat5gBYuOEk7WYMYkq5uh8jDx0o/edit?usp=sharing (deadline 19 March)

THINGS TO INVESTIGATE
-------------
- Model architectures (AttentiveFP vs ChemProp) (low priority)
- Make doc pretty (add refs) (Carl) (high priority)
- HP optimization (Together?, Carl)
- Analyze attentions (Enrico)
- Chemical space analysis (Carl, Riccardo) --> generalize to new molecules? 
- Compare MTL vs STL vs MTL with same mols from STL (Enrico + Riccardo)

May 6th 2024
- Which properties? And how to combine?
- Chemical space analysis
- Make doc pretty 
- HP optimization
- Compare MTL vs STL


