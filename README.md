# Reducing Divergences for Generalized Federated Learning via Heterogeneity-aware Parameter Guidance 

<p align="center">
  <a href="#"><img alt="Python3" src="https://img.shields.io/badge/Python-3-brightgreen.svg?style=flat-square"></a>
  <a href="#"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%3E=1.11-orange"></a>
</p>


This repository is the official implementation of [**Reducing Divergences for Generalized Federated Learning via Heterogeneity-aware Parameter Guidance**]. (Submitted to **NeurIPS 2025**)
<img src="./imgs/nips_framework.jpg" width = "80%" height = "" alt="FedRD" TITLE="Overview of FedRD." />


## Paper of our work

Reducing Divergences for Generalized Federated Learning via Heterogeneity-aware Parameter Guidance: [paper link](https://openreview.net/forum?id=fhkgIxJUF0)
![image](./imgs/feature.jpg)



## Baselines
Following the implementation of work FedGA, we here provide four baseline menthod implementations in ./algorithms/:
 - FedAVG [https://proceedings.mlr.press/v54/mcmahan17a]
 - FedProx [https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html]
 - Scaffold [https://proceedings.mlr.press/v119/karimireddy20a]
 - FedGA [https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Federated_Domain_Generalization_With_Generalization_Adjustment_CVPR_2023_paper.pdf]



## Requirements

- Python 3.9.7
- numpy 1.20.3
- torch 1.11.0
- torchvision 0.12.0


To install requirements:
```
pip install -r requirements.txt
```

## Dataset

To implement datasets mentioned in the paper, first create directory for log files and change the dataset path (`pacs_path`,`officehome_path`, and `DomainNet`) and log path (`log_count_path`) in configs/default.py.
Please download the datasets from the official links:

- [PACS](https://domaingeneralization.github.io)
- [OfficeHome](https://hemanthdv.github.io/officehome-dataset)
- [DomainNet](http://ai.bu.edu/M3SDA/)


## Training from scratch

We release the code for PACS dataset and officehome dataset can be applied by only changing the dataloader_obj in data/{officehome, DomainNet}_dataset.py. 

Then running the code:

`
python ./algorithms/fedrd_train.py --test_domain p --lr 0.001 --batch_size 16 --comm 40 --model resnet18 --note debug
`

## Acknowledgement

Part of our code is borrowed from the following repositories.

- FedGA [https://github.com/MediaBrain-SJTU/FedDG-GA]
- FACT [https://github.com/MediaBrain-SJTU/FACT]
- DomainBed [https://github.com/facebookresearch/DomainBed]
- FedNova [https://github.com/JYWa/FedNova]
- SCAFFOLD-PyTorch [https://github.com/KarhouTam/SCAFFOLD-PyTorch]
We thank to the authors for releasing their codes. Please also consider citing their works.
