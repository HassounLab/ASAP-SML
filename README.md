# ASAP-SML: An Antibody Sequence Analysis Pipeline Using Statistical Testing and Machine Learning

Antibody Sequence Analysis Pipeline Using Statistical Testing and Machine Learning (ASAP-SML) is a pipeline to identify distinguishing features in targeting antibody set when compared to a reference non-targeting set. The pipeline first extracts germline, CDR canonical structure, isoelectric point and frequent positional motifs features from sequences and creates an antibody feature fingerprint. Machine-learning and statistical significance testing are applied to antibody sequences and feature fingerprints to identify distinguishing feature values and combinations thereof. When applied to an MMP-targeting set, ASAP identifies salient features and recommends features to use when designing novel MPP-targeting antibody sequences.

## How to install
### Requirements: 
An [Anaconda python environment](https://www.anaconda.com/download) is recommmended.
Check the environment.yml file, but primarily:
- python >= 3.5
- pandas
- graphviz
- jupyter
- numpy
- scikit-learn
- scipy
- biopython

Jupyter notebook is required to run the ipynb examples.

### via Anaconda 
We recommend installing using Anaconda as follows:
```
conda create --name asap --file enviroment.yml
source activate asap
```

## Example: Matrix Metalloproteinases (MMP) targeting and reference antibody sequence set

This repository contains an example of how to run the ASAP pipeline on the MMP-targeting and reference antibody sequence set.

To run the script, open the terminal and go to the project directory, then run:

`
jupyter notebook
`

Take a look at the file "ASAP.ipynb". Parameters are set based on the users choice. Once you have set the parameters, run the notebook document step-by-step (one cell a time) by 

- Pressing shift + enter

Or, run the whole notebook in a single step by 

- Clicking on the menu Cell -> Run All.

## Components
ASAP.ipynb : main script for running ASAP pipeline 

- **./ASAP/FeatureExtraction.py** -  functions for feature extraction on Chothia numbered antibody sequences.
- **./ASAP/SequenceAndFeatureAnalysis.py** - functions for sequence and feature analysis on antibody sequences. 
- **./ASAP/DesignRecommendation.py** - functions to generate design recommendation trees for specific targeting antibody sequences.

## Data

- Data to run ASAP: [BLOSUM-62 substitution matrix](https://en.wikipedia.org/wiki/BLOSUM#cite_ref-henikoff_1-0) and [Canonical Structure Definition](http://circe.med.uniroma1.it/pigs/canonical.php)

- Data to run ASAP on MMP-targeting example: MMP-targeting and reference set. 

MMP-targeting set is composed of publicly available antibody sequence data. Reference set is from the Protein Data Bank (PDB) and it consists of human and murine antibody sequences that do not bind or inhibit MMPs. Please see our paper for details.

## Authors:
This software is written by Xinmeng Li, James Van Deventer, Soha Hassoun (Soha.Hassoun@tufts.edu). 

Publication: ["ASAP-SML: An Antibody Sequence Analysis Pipeline Using Statistical Testing and Machine Learning"](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007779)

**Please cite our work:**

Li, Xinmeng, James A. Van Deventer, and Soha Hassoun. "ASAP-SML: An antibody sequence analysis pipeline using statistical testing and machine learning." PLoS computational biology 16.4 (2020): e1007779.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

