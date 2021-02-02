# The MECCANO Dataset: Understanding Human-Object Interactions from Egocentric Videos in an Industrial-like Domain

This is the related official github repo associated to the MECCANO Dataset.

![alt text]https://github.com/francescoragusa/MECCANO/blob/master/images/MECCANO.png?raw=true)

The MECCANO Dataset is the first dataset of egocentric videos to study human-object interactions in industrial-like settings. You can download the MECCANO dataset from the [project web page](https://iplab.dmi.unict.it/MECCANO/).

#### Use the MECCANO Dataset with PySlowFast
To use the MECCANO Dataset in PySlowfast please follow the instructions below:

* Install PySlowFast following the [official instructions](https://github.com/facebookresearch/SlowFast/blob/master/INSTALL.md);
* Download the PySlowFast_files folder from this repository;
* Place the files "__init__.py", "meccano.py" and "sampling.py" in your slowfast/datasets/ folder.

Now, run the training/test with:
```
python tools/run_net.py --cfg path_to_your_config_file --[optional flags]
```
#### Citing the MECCANO Dataset
If you find the MECCANO Dataset useful in your research, please use the following BibTeX entry for citation.
```BibTeX
@inproceedings{ragusa2020meccano,
  title = { The MECCANO Dataset: Understanding Human-Object Interactions from Egocentric Videos in an Industrial-like Domain },
  author = { Francesco Ragusa and Antonino Furnari and Salvatore Livatino and Giovanni Maria Farinella },
  year = { 2021 },
  eprint = {  2010.05654  },
  booktitle = { IEEE Winter Conference on Application of Computer Vision (WACV) },
  primaryclass = {  cs.CV  },
  url = {  https://iplab.dmi.unict.it/MECCANO  },
  pdf = {  https://arxiv.org/pdf/2010.05654.pdf  },
}
```
