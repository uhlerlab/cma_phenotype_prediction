# Cross-modal autoencoder phenotype prediction 
Code for phenotype prediction from cross-modal autoencoder embeddings used in the following [paper](https://www.biorxiv.org/content/10.1101/2022.05.26.493497v1).

All dependencies are provided in the environment.yml file, which can be used to create a conda environment.  

## Description: 

main.py requires the following inputs (described in options_parser.py): 

1. Model embeddings 
2. Phenotypes  
3. Indices for train, validation, and test samples 
4. Choice of kernel (ntk, linear, laplace), 
5. Number of epochs for kernel regression

main.py will write the $R^2$ values for kernel regression in an output directory (csv_outputs).  

## Format for input files: 

1. Model embeddings should be provided in a tsv format.  Our code uses 'sample_id' as a key to identify each MRI/ECG sample and each row has a tab separated list of real values representing coordinates of latent dimensions. 
2. Phenotypes should be provided in a tsv format.  Our code uses 'sample_id' to link phenotypes to corresponding embeddings and each row has a tab separated list of real values for phenotypes (e.g. LVM, LVEDV, etc.).  
3. Train, validation, and test samples should be sets (not lists) of sample_ids used to identify data for training, validation, and test.  
