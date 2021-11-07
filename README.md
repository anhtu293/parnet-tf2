# ParNet-TF2
- Tensorflow implementation of Parnet.
- Paper: https://arxiv.org/pdf/2110.07641.pdf

# How to run
## Build & run image docker
```
cd parnet-tf2
docker build docker/. -t parnet-tf2
nvidia-docker run -it --rm -v ~/Documents/parnet-tf2:/workspace parnet-tf2 bash
```

## Train
```
python train.py --dataset cifar10 --scale large --batch-size 16 --gpus 0,1 --output workdirs --epochs 100
```
- Dataset: imagenet - cifar10 - cifar100.
    - Model architecture for cifar is different from imagenet due to the difference in image resolution. All datasets used in this code are ready-to-use datasets from tensorflow_datasets.
    - If you use your own dataset, you should implement a data generator, or use tf.data.Dataset:
```
class DatGenerator(tf.keras.utils.Sequence):
    def __init__(self, **kwargs):
        super(dataGenerator, self).__init__(**kwargs)
        ...
    def __len__(self):
        ...
    def __getitem__(self, idx):
        ...
```
- Scale: small - medium - large - extra.
- Gpus: multi-gpu training supported.
- Output: path to save model checkpoints.
## Re-param ParNet Block
```
python reparam.py --dataset cifar10 --checkpoint workdirs/checkpoint.hdf5 --output new_model
```
- Re-param method is introduced in RepVGG (https://arxiv.org/pdf/2101.03697.pdf).
- The result of original and reparam-ed model are the same.
## Evaluate
```
python evaluate.py --dataset cifar10 --batch-size 32 --checkpoint new_model.hdf5
```
- Attention: evaluation code uses reparam-ed model.
- Dataset is the same as in training.


# Updates:
- [ ] Add pretrained weights
- [x] Add a ready-to-use notebook
- [x] Add evaluation code
- [x] Add reparam function and script
- [x] Add training script on cifar and imagenet with multigpu support.