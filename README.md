# Video Captioing.

This repo implement the code for video captioning, based on the paper ["Joint Localizing and Describing Events for Dense Video Captioning (CVPR 2018)"](https://arxiv.org/abs/1804.08274).


## Requirements

Python 2.7

PyTorch 0.4.0


## Pretrained models

Will be updated later.

## Dataset

The ActivityNet annotation dataset can be download from [Activity Net homepage](http://activity-net.org/download.html).
And ActivityNet caption dataset and C3D video features also can be download from [Dense-Captioning Events in Videos](https://cs.stanford.edu/people/ranjaykrishna/densevid/).

When you download all datasets, put activity_net.v1-3.min.json file in the folder data/ , put feature file in the folder data/actnet/, and unzip captions.zip in the folder data/.

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

### Preprocessing

Before training the model, you need to preprocess the dataset.

```
python prepro_atts.py
python prepro_caps.py
```

The first is the dataset preprocessing for attribute detector.
The second is the dataset preprocessing used to train the sentence generator and dense video captioning.

### Training for attribute detector

```
python run_detector.py
```

### Training for sentence generator

Use the weight of the attribute detector trained in the previous step.

```
python run_sent_gen.py --resume-att {attribute detector weight file}
```

### Training for dense video captioning

Use the weight of the attribute detector and sentence generator trained in the previous step.

```
python run_dvc_test.py --file-name {output file name} --resume-att {attribute detector weight file} --resume-sg {sentence generator weight file}
```