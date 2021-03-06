# The MECCANO Dataset

This is the official github repository related to the MECCANO Dataset.

<div align="center">
  <img src="images/MECCANO.gif"/>
</div>

The MECCANO Dataset is the first dataset of egocentric videos to study human-object interactions in industrial-like settings. You can download the MECCANO dataset and its annotations from the [project web page](https://iplab.dmi.unict.it/MECCANO/).

## Use the MECCANO Dataset with PySlowFast
To use the MECCANO Dataset in PySlowfast please follow the instructions below:

* Install PySlowFast following the [official instructions](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md);
* Download the PySlowFast_files folder from this repository;
* Place the files "__init__.py", "meccano.py" and "sampling.py" in your slowfast/datasets/ folder.

Now, run the training/test with:
```
python tools/run_net.py --cfg path_to_your_config_file --[optional flags]
```

## Use the MECCANO Dataset with Detectron2
To use the MECCANO Dataset in Detectron2 to perform Object Detection and Recognition please follow the instructions below:

* Install Detectron2:
    ```
    pip install -U torch torchvision cython
    pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    git clone https://github.com/facebookresearch/detectron2 detectron2_repo
    pip install -e detectron2_repo
    # You can find more details at https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
    ```
* Register the MECCANO Dataset adding the following instructions in detectron2_repo/tools/run_net.py, in the main() function:
    ```
    register_coco_instances("Meccano_objects_train", {}, "/path_to_your_folder/instances_meccano_train.json", "/path_to_the_MECCANO_active_object_annotations_frames/")
    register_coco_instances("Meccano_objects_val", {}, "/path_to_your_folder/instances_meccano_val.json", "/path_to_the_MECCANO_active_object_annotations_frames/")
    register_coco_instances("Meccano_objects_test", {}, "/path_to_your_folder/instances_meccano_test.json","/path_to_the_MECCANO_active_object_annotations_frames/")
    ```

Now, run the training/test with:
```
python tools/train_net.py --config-file path_to_your_config_file --[optional flags]
```

## Model Zoo and Baselines

### PySlowFast models

We provided pretrained models on the MECCANO Dataset for the action recognition task:
| architecture | depth |  model  | config |
| ------------- | -------------| ------------- | ------------- |
| C2D | R50 | [`coming soon`](https://iplab.dmi.unict.it/MECCANO/) | configs/action_recognition/C2D_8x8_R50.yaml |
| I3D | R50 | [`coming soon`](https://iplab.dmi.unict.it/MECCANO/) | configs/action_recognition/I3D_8x8_R50.yaml |
| SlowFast | R50 | [`link`](https://iplab.dmi.unict.it/MECCANO/models/SLOWFAST_8x8_R50_MECCANO.pyth) | configs/action_recognition/SLOWFAST_8x8_R50.yaml |

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
