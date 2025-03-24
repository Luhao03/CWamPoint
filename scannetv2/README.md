
## Prepare

Download ScanNetV2 dataset: http://www.scan-net.org/#code-and-data

Run script:

```shell
python prepare_dataset.py -i /home/user/Datasets/ScanNetV2/ -o scannetv2-preprocessed

```

The `scannetv2-preprocessed` will be saved in `/home/user/Datasets/`. 

(Alternative) Our preprocessed data can be  downloaded [here](https://drive.google.com/file/d/1tp3J2uHYs3QM29VXgY7mRPqiyyyF_kzp/view?usp=sharing), please agree the official license before download it.

Link the dataset:

```shell
ln -s /home/user/Datasets/scannetv2-preprocessed/ dataset_link
```

## Train

Run train script:

```shell
# train a small size model
python train.py

# train a large size model, experiment named 'test', train log output into './exp/'
python train.py --exp test --model_size l

# batch size is 6, and use checkpoint method to save memory(optional)
python train.py --exp test --model_size l --batch_size 6 --use_cp

# train a custom size model, experiment named 'test'
vim configs/scannetv2_custom.py
python train.py --exp test --model_size c

# more usage help
python train.py -h 
```

## Finetune or Resume

Run train script:

```shell
# resume the last experiment named 'test', for example 'test-001'
python train.py --exp test --model_size l --mode resume
# load the pretrained model, experiment named 'test'
python train.py --exp test --model_size l --mode finetune --ckpt <checkpoint_file>
```

## Test

Run test script:

```shell
# test a large size model, test log output into './exp-test/'
python test.py --model_size l --ckpt <checkpoint_file>
```

## Visual

Run test script:

```shell
# test a large size model and save the visual results into './visual/'
python test.py --model_size l --ckpt <checkpoint_file> --vis
```

Run visual script:

```shell
python visual.py
```

