# The MECCANO Dataset

This is the official github repository related to the MECCANO Dataset.

<div align="center">
  <img src="images/MECCANO.gif"/>
</div>

The MECCANO Dataset is the first dataset of egocentric videos to study human-object interactions in industrial-like settings. You can download the MECCANO dataset from the [project web page](https://iplab.dmi.unict.it/MECCANO/).

## Use the MECCANO Dataset with PySlowFast
To use the MECCANO Dataset in PySlowfast please follow the instructions below:

* Install PySlowFast following the [official instructions](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md);
* Download the PySlowFast_files folder from this repository;
* Place the files "__init__.py", "meccano.py" and "sampling.py" in your slowfast/datasets/ folder.

Now, run the training/test with:
```
python tools/run_net.py --cfg path_to_your_config_file --[optional flags]
```


## Model Zoo and Baselines

### PySlowFast models

We provided pretrained models on the MECCANO Dataset for the action recognition task:
| architecture | depth |  model  | config |
| ------------- | -------------| ------------- | ------------- |
| C2D | R50 | [`link`](https://iplab.dmi.unict.it/MECCANO/) | configs/action_recognition/C2D_8x8_R50.yaml |
| I3D | R50 | [`link`](https://iplab.dmi.unict.it/MECCANO/) | configs/action_recognition/I3D_8x8_R50.yaml |
| SlowFast | R50 | [`link`](https://iplab.dmi.unict.it/MECCANO/) | configs/action_recognition/SLOWFAST_8x8_R50.yaml |

### Detectron2 models

We provided pretrained models on the MECCANO Dataset for the active object recognition task:
| architecture | depth |  model  | config |
| ------------- | -------------| ------------- | ------------- |
| Faster RCNN | R101_FPN | [`link`](https://iplab.dmi.unict.it/MECCANO/models/model_meccano_active_objects.pth) | configs/active_object_recognition/meccano_active_objects.yaml |

## Citing the MECCANO Dataset
If you find the MECCANO Dataset useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@inproceedings{ragusa2021meccano,
  title = {The MECCANO Dataset: Understanding Human-Object Interactions from Egocentric Videos in an Industrial-like Domain},
  author = {Francesco Ragusa and Antonino Furnari and Salvatore Livatino and Giovanni Maria Farinella},
  year = {2021},
  eprint = {2010.05654},
  booktitle = {IEEE Winter Conference on Application of Computer Vision (WACV)}
}
```
