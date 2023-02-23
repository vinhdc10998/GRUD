# GenotypeImputationGRU

## Overview

This repository contains a Python implementation of GRUD model, which is a genotype imputation based on deep learning algorithms. In specific, GRUD is composed of two components: *Generator* and *Discriminator*. In addition to, our approach resolves the reference panel privacy problem while improving both the accuracy and running time.

If you have any feature requests or questions, feel free to leave them as GitHub issues!
## REQUIREMENT

- Python = 3.8.8
- Python packages:
  - NumPy
  - Scikit-learn
  - PyTorch

```script
pip install -r requirements.txt
```

## EXAMPLE USAGE

### Dataset
In this repository, the example dataset which is used to train and evaluation is 1000 Genome Project (1KGP) phase3 integrated dataset for chromosome22.
- Type of input data:
  - Phased genotype with HAP/LEGEND format ( For details, please see https://github.com/kanamekojima/rnnimp)
  - Weighted of trained model
- Type of output data:
  - Genotype imputation results in Oxford GEN format

### Imputation
For example, GRUD model imputes genotypes for small regions (1-10) separately

```script
python eval.py  --root-dir <path of LENGEND/HAP files> \
                --model-config-dir <path of config model files> \
                --model-type dis \
                --batch-size 128 \
                --regions 1-10 \
                --model-dir <path of weighted model files> \
                --gpu 2 \ 
                --result-gen-dir <path of output files> \
                --best-model
```
## Arguments
| Args | Default | Description |
| :--- | :--- | :--- |
| --model-type STR | None | Type of model |
| --root-dir STR | None | Data folder |
| --model-config-dir STR | None | Config model folder |
| --gpu STR | None | GPU's Id |
| --batch-size INT | 2 | Type of model |
| --regions STR | 1 | Range of regions |
| --chr STR | chr22 | Chromosome |
| --model-dir STR | model/weights | weight model folder |
| --result-gen-dir STR | results/ | result folder |
| --dataset STR| None | Custom dataset |

## CONTACT
Developer: Duong Chi Vinh, GeneStory

E-mail: vinh.duong [AT] genestory [DOT] ai
