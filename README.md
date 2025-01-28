# Few Edges Are Enough: Few-Shot Network Attack Detection with Graph Neural Networks

Code of the [FEAE paper](https://link.springer.com/chapter/10.1007/978-981-97-7737-2_15) (IWSEC'24 best paper), showing the impact of integrating few-shot attack examples within the training of self-supervised detection models based on GNNs.

## Datasets

Simply download `NF-CSE-CIC-IDS2018-v2` and `NF-UNSW-NB15-v2` files at this [URL](https://staff.itee.uq.edu.au/marius/NIDS_datasets/). Unzip the archives and move `NF-CSE-CIC-IDS2018-v2.csv` and `NF-UNSW-NB15-v2.csv` from the data folder to the root of the repo.

## Installation

This project uses DGL and PyTorch.

### Installation with CUDA 12.1

```
conda create -n feae python=3.9
conda activate feae
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c dglteam/label/th23_cu121 dgl
pip install pydantic matplotlib
conda install -c conda-forge category_encoders
pip install scikit-learn==1.5.2
```

## Run FEAE experiments

#### On the NF-CSE-CIC-IDS2018-v2 dataset
```
python few_shot_dgi.py --dataset=CSE_CIC --pos-augment=augment_drop_nodes --k=1 --mlp-lr=0.0001
```
> Training may take up to 30 min.

#### On the NF-UNSW-NB15-v2 dataset
```
python few_shot_dgi.py --dataset=UNSW --pos-augment=augment_drop_nodes --k=1 --mlp-lr=0.001
```

## Run baseline experiments
#### Supervised baselines
```
python line_graph_bench.py --encoder=[LineGCN | LineGAT | LineSAGE] --dataset=CSE_CIC
```

```
python egraphsage_supervised_bench.py
```

## License

See [license](LICENSE).
