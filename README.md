Deep Learning in Low-Data Regime: how multi-task learning can improve predictions. 
-------------
With this work, we show how multi-task learning can improve the quality of predictions in low-data regime, by learning from related properties. The example case studied here is the partition coefficient of a compound between octanol and water (logP). We show how in low data regime, both on internal and external test data, multi-task learning performs better then single-task learning in logP prediction. Additionally, we show that picking diverse compounds for the training set improved performance over random undersampling. 

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


Contributors (alphabetically)
-------------
Domen Preceljc, Enrico Ruijsenaars, Carl Schiebroek, Riccardo Solazzo, Jiayi Zhu
