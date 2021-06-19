# TRAIN

Data use t train

All description must expend how and why you create this also cite if it have

## Folder tree should have structur

Same as **Folder tree should have structure** at test folder.

## Truth structure

```tree
train
├── G1K_chr22_biallelic_train.log                   - process log file.
├── G1K_chr22_biallelic_train.recode.erate          - was created when run script create "m3vcf" file.
├── G1K_chr22_biallelic_train.recode.m3vcf.gz       - was created by minimac3. See at description below.
├── G1K_chr22_biallelic_train.recode.rec            - was created when run script create "m3vcf" file.
├── G1K_chr22_biallelic_train.recode.vcf.gz         - data for training. See at description below.
└── README.md
```

## Description

**NOTE**: see **README.md** at data root folder to use minimac.

**G1K_chr22_biallelic_train.recode.vcf.gz** file was created by this script at root project:

```script
vcftools --gzvcf ./data/interim/G1K_chr22_biallelic.vcf.gz --remove ./data/external/test_100_samples.txt --out ./data/train/G1K_chr22_biallelic_train --recode --recode-INFO-all
gzip ./data/train/G1K_chr22_biallelic_train.recode.vcf.gz
```

**G1K_chr22_biallelic_train.recode.m3vcf.gz** file was created by this script at this folder:

```script
minimac3 --refHaps G1K_chr22_biallelic_train.recode.vcf.gz --processReference --prefix G1K_chr22_biallelic_train.recode
```
