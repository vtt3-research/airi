# Video Captioning

This repo implement the code for video captioning, based on the paper [Joint Localizing and Describing Events for Dense Video Captioning](https://arxiv.org/abs/1804.08274) (CVPR 2018).


## Requirements

- Python 2.7
- PyTorch 0.4.0

python packages
- h5py
- json
- numpy


## Pretrained models

...

## Dataset

The ActivityNet annotation dataset can be download from [Activity Net homepage](http://activity-net.org/download.html).
And ActivityNet caption dataset and C3D video features also can be download from [Dense-Captioning Events in Videos](https://cs.stanford.edu/people/ranjaykrishna/densevid/).

When you download all datasets, put `activity_net.v1-3.min.json` file in the folder `data/` , put feature file in the folder `data/actnet/`, and unzip `captions.zip` in the folder `data/`.

Finally, the files are organized as follows.

```
data/
-- actnet/
---- sub_activitynet_v1-3.c3d.hdf5
-- captions/
---- test_ids.json
---- train.json
---- train_ids.json
---- val_1.json
---- val_2.json
---- val_ids.json
-- activit_net.v1-3.min.json
```

## Training

You can input various argument options that are not listed below.
If you want to use another options, see the argument part in each file.

### 1. Preprocessing

Before training the model, you need to preprocess the dataset.

```
python prepro_atts.py
python prepro_caps.py
```

The first is the dataset preprocessing for attribute detector.
The second is the dataset preprocessing used to train the sentence generator and dense video captioning.

### 2. Training for attribute detector

```
python run_detector.py --file-name {output file name}
```

### 3. Training for sentence generator

Use the weight of the attribute detector trained in the previous step.

```
python run_sent_gen.py --file-name {output file name} --resume-att {attribute detector weight file}
```

### 4. Training for dense video captioning

Use the weight of the attribute detector and sentence generator trained in the previous step.

```
python run_dvc.py --file-name {output file name} --resume-att {attribute detector weight file} --resume-sg {sentence generator weight file}
```

## For another dataset

### MSR-VTT dataset

1. Prepare data files

Download the dataset from [MSRVTT homepage](ms-multimedia-challenge.com/2016/dataset) and prepare features from MSRVTT video set using C3D model

Features file type is hdf5 and contain video name, feature vector pairs as follows.

```
features = h5py.File('features.hdf5', 'r')

# video name : video0
feat = features['video0'].value
```

Data files are organized as follows.

```
data/
-- MSRVTT/
---- videodatainfo.json
---- msrvtt_features.hdf5
```

2. Preprocessing

Preprocess dataset.

```
python prepro_msrvtt.py
```

3. Training

Training the attribute detector model.

```
python run_detector.py --root data/MSRVTT --file-name {output file name} --feature-dim {MSR-VTT feature dimension} --num-class 20
```

And training the sentence generator model using trained attribute detector model.

```
python run_sg_msrvtt.py --file-name {output file name} --resume-att {attribute detector weight file} --feature-dim {MSR-VTT feature dimension}
```

If you want to use reinforcement learning (Self-Critical) additionally, input rl-flag option.

```
python run_sg_msrvtt.py --file-name {output file name} --resume-att {attribute detector weight file} --resume-sg {sentence generator weight file above} --feature-dim {MSR-VTT feature dimension} --rl-flag
```

For each epoch, automatically evaluate metric with (current) trained model for validation set.
