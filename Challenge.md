# Multimodal Action Recognition on the MECCANO Dataset ([ICIAP Competition with Prize!](https://iplab.dmi.unict.it/MECCANO/challenge.html))

Submission deadline has been extended to August 07, 2023!!

This is the official github repository related to the MECCANO Dataset.

<div align="center">
  <img src="images/MECCANO_Multimodal.gif"/>
</div>

MECCANO is a multimodal dataset of egocentric videos to study humans behavior understanding in industrial-like settings. The multimodality is characterized by the presence of gaze signals, depth maps and RGB videos acquired simultaneously with a custom headset. You can download the MECCANO dataset and its annotations from the [project web page](https://iplab.dmi.unict.it/MECCANO/).

## Use the MECCANO Dataset with PySlowFast
To use the MECCANO Dataset in PySlowfast please follow the instructions below:

* Install PySlowFast following the [official instructions](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md);
* Download the PySlowFast_files folder from this repository;
* Place the files "__init__.py", "meccano.py" and "sampling.py" in your slowfast/datasets/ folder;
* Place the files "__init__.py", "custom_video_model_builder_MECCANO_gaze.py" in your slowfast/models/ folder (to use the gaze signal).

Now, run the training/test with:
```
python tools/run_net.py --cfg path_to_your_config_file --[optional flags]
```
### Pre-Extracted Features
We provide pre-extracted features of MECCANO Dataset:

* RGB features extracted with SlowFast: [`coming soon`]

## Model Zoo and Baselines

### Multimodal Action Recognition

#### PySlowFast models

We provided pretrained models on the MECCANO Dataset for the action recognition task (only for the first version of the dataset):
| architecture | depth |  model  | config |
| ------------- | -------------| ------------- | ------------- |
| I3D | R50 | [`link`](https://iplab.dmi.unict.it/sharing/MECCANO/models/action_recognition/first_version/I3D_8x8_R50_MECCANO.pyth) | configs/action_recognition/I3D_8x8_R50.yaml |
| SlowFast | R50 | [`link`](https://iplab.dmi.unict.it/sharing/MECCANO/models/action_recognition/first_version/SLOWFAST_8x8_R50_MECCANO.pyth) | configs/action_recognition/SLOWFAST_8x8_R50.yaml |

We provided pretrained models on the MECCANO Multimodal Dataset for the action recognition task:
| architecture | depth | modality | model  | config |
| ------------- | ------------- | -------------| ------------- | ------------- |
| SlowFast | R50 | RGB | [`link`](https://iplab.dmi.unict.it/sharing/MECCANO/models/action_recognition/SLOWFAST_8x8_R50_RGB_MECCANO.pyth) | configs/action_recognition/SLOWFAST_8x8_R50_MECCANO.yaml |
| SlowFast | R50 | Depth | [`link`](https://iplab.dmi.unict.it/sharing/MECCANO/models/action_recognition/SLOWFAST_8x8_R50_Depth_MECCANO.pyth) | configs/action_recognition/SLOWFAST_8x8_R50_MECCANO.yaml |

## Citing the MECCANO Dataset
If you find the MECCANO Dataset useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@misc{ragusa2022meccano,
title={MECCANO: A Multimodal Egocentric Dataset for Humans Behavior Understanding in the Industrial-like Domain},
author={Francesco Ragusa and Antonino Furnari and Giovanni Maria Farinella},
year={2022},
eprint={2209.08691},
archivePrefix={arXiv},
primaryClass={cs.CV}
}
```
Additionally, cite the original paper:
```BibTeX
@inproceedings{ragusa2021meccano,
  title = {The MECCANO Dataset: Understanding Human-Object Interactions from Egocentric Videos in an Industrial-like Domain},
  author = {Francesco Ragusa and Antonino Furnari and Salvatore Livatino and Giovanni Maria Farinella},
  year = {2021},
  eprint = {2010.05654},
  booktitle = {IEEE Winter Conference on Application of Computer Vision (WACV)}
}
```
