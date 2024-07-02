# A rapid and reference-free imputation method for low-cost genotyping platforms
![Screenshot](image/GRUD.svg)

This repository contains a Python implementation of GRUD model, which is a genotype imputation based on deep learning algorithms. In specific, GRUD is composed of two components: *Generator* and *Discriminator*. The generator model undertakes a mission to create tokens, and the discriminator tries to verify tokens created by the generator model. In the current study, we assumed that unobserved variants are tokens in natural languages which would be predicted from a known paragraph of observed variants. In addition to, our approach resolves the reference panel privacy problem while improving both the accuracy and running time. 

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

### 1. Dataset
In this repository, the example dataset that is used to train and evaluate is the 1000 Genome Project (1KGP) phase 3 integrated dataset for chromosome22.

Download: https://drive.google.com/drive/folders/17RIdXWoIKJsxjzfYmSmvW19kaQHxoafC?usp=sharing
- Type of input data:
  - Phased genotype with HAP/LEGEND format ( For details, please see https://github.com/kanamekojima/rnnimp)
  - Weighted of trained model
- Type of output data:
  - Genotype imputation results in Oxford GEN format
  - VCF format

### 2. Evaluation/ Inference
For example, GRUD model imputes genotypes for small regions (1-10) separately. 

In evaluation, the root directory should contain ground truth files while inference doesn't need this.

```bash
python eval.py  --root-dir <path of LEGEND/HAP files> \
                --model-config-dir <path of config model files> \
                --model-dir <path of weighted model files> \
                --result-gen-dir <path of output files> \
                --sample <path of sample name file> \
                --regions 1-10 \
                --best-model
```

### 3. Training
We provides small regions of chromosome 22 for training 1KGP data.

```bash
python train.py --root-dir <path of LEGEND/HAP files> \
                --model-type dis \
                --model-config-dir <path of config model files> \
                --batch-size 128 \
                --epochs 100 \
                --chr chr22 \
                --lr 0.001 \
                --output-model-dir <output path of weighted model files> \
                --early-stopping \
                --gpu 0 \
                --region 1-10
```

### 5. Arguments
| Args | Default | Description |
| :--- | :--- | :--- |
| --model-type STR | dis | Type of model |
| --root-dir STR | None | Data folder |
| --model-config-dir STR | None | Config model folder |
| --gpu STR | None | GPU's Id |
| --batch-size INT | 128 | Type of model |
| --regions STR | 1 | Range of regions |
| --chr STR | chr22 | Chromosome |
| --model-dir STR | model/weights | weight model folder |
| --result-gen-dir STR | results/ | result folder |
| --dataset STR| None | Custom dataset |
| --lr FLOAT| 1e-4 | learning rate |
| --output-model-dir STR| None | Custom dataset |
| --early-stopping BOOL | False | Early stopping |
| --best-model BOOL| False | Get best model to test |
| --sample STR| None | path to sample name file |

## TODO
- [x] Training model source code 
- [x] Evaluation model source code
- [x] Inference model source code (updated on 07/01/2024)
- [x] Parallel in inference (updated on 07/02/2024)
- [ ] Preprocess data source code

## CITATION
If you find GRUD or any of the scripts in this repository useful for your research, please cite:

> Chi Duong, V., Minh Vu, G., Khac Nguyen, T., Tran The Nguyen, H., Luong Pham, T., S. Vo, N., & Hong Hoang, T. (2023). A rapid and reference-free imputation method for low-cost genotyping platforms. Scientific Reports, 13(1), 23083. https://doi.org/10.1038/s41598-023-50086-4

## LICENSE
The scripts in this repository are available under the MIT License. For more details, see the LICENSE file.

## CONTACT
Developer: Duong Chi Vinh, GeneStory

E-mail: vinh.duong [AT] genestory [DOT] ai
