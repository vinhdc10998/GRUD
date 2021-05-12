# GenotypeImputationGRU

## OVERVIEW

GenotypeImputationGRU is a recurrent neural network based genotype imputation program implemented in Python. 

## REQUIREMENT

- Python >= 3.5
- Python packages
  - NumPy
  - PyTorch

```script
pip install -r requirements.txt
```

## RUN

### Example train

```script
python train.py --root-dir data/org_data/\
                --model-config-dir model/config/\
                --epochs 10\
                --batch-size 4\
                --region 1\
                --model-type Higher
```

### Example eval

```script
python eval.py  --root-dir data/test_100_sample_GT/\
                --model-config-dir model/config/\
                --model-type Lower\
                --batch-size 32\
                --regions 1\
                --model-dir model/weights/
```
