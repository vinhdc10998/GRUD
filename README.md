# GenotypeImputationGRU

## Example train
```
python train.py --root-dir data/org_data/\
                --model-config-dir model/config/\
                --epochs 10\
                --batch-size 4\
                --region 1\
                --model-type Higher
```

### Example eval
```
python eval.py  --root-dir data/test_100_sample_GT/\
                --model-config-dir model/config/\
                --model-type Lower\
                --batch-size 32\
                --regions 1\
                --model-dir model/weights/
```
