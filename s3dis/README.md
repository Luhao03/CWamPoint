
## Prepare

Download S3DIS dataset `Stanford3dDataset_v1.2_Aligned_Version`: http://buildingparser.stanford.edu/dataset.html

Run script:

```shell
python prepare_dataset.py -i /home/user/Datasets/Stanford3dDataset_v1.2_Aligned_Version/ -o s3dis-preprocessed

```
Solve data error:
> ValueError: the number of columns changed from 6 to 5 at row 180389; use `usecols` to select a subset and avoid this error

```shell


vim +180389 /home/user/Datasets/Stanford3dDataset_v1.2_Aligned_Version/Area_5/hallway_6/Annotations/ceiling_1.txt

```

The `s3dis-preprocessed` will be saved in `/home/user/Datasets/`. 

(Alternative) Our preprocessed data can be  downloaded [here](https://drive.google.com/file/d/1h2RXcKZk-yr6f5J_3FBkBxyBayg8zRA8/view?usp=sharing), please agree the official license before download it.

Link the dataset:

```shell
ln -s /home/user/Datasets/s3dis-preprocessed/ dataset_link
```

## Train

Run train script:

```shell
# train a small size model
python train.py

# train a large size model, experiment named 'test', train log output into './exp/'
python train.py --exp test --model_size l

# train a custom size model, experiment named 'test'
vim configs/s3dis_custom.py
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

