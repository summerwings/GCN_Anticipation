# Towards Graph Representation Learning Based Surgical Workflow Anticipation

The current GitHub page for this paper is (https://github.com/FrancisXZhang/GCN_Anticipation). If you encounter any issues with my previous projects, please refer to this page.

Example Code for the paper Towards Graph Representation Learning Based Surgical Workflow Anticipation

![image](Network_Overview.png)

## Environment Setup
First please create an appropriate environment using conda: 

> conda env create -f surgery.yaml
> 
> conda activate surgery

## Test Pre-Trained Models
Evaluate on Sample dataset:
> python main_infer.py


## Train a Model
Train on Sample dataset:
> python main.py

In training, our default stream is only based on graph-level information, as we proposed in our paper.

## Citing

If you find this work useful, please consider our paper to cite:

```
@inproceedings{zhang22towards,
 author={Zhang, Francis Xiatian and Moubayed, Noura Al and Shum, Hubert P. H.},
 booktitle={Proceedings of the 2022 IEEE-EMBS International Conference on Biomedical and Health Informatics},
 title={Towards Graph Representation Learning Based Surgical Workflow Anticipation },
 year={2022},
 publisher={IEEE},
 location={Ioannina, Greece},
}
```
