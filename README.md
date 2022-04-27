# Task-Aware Active Learning for Endoscopic Image Analysis

Official PyTorch implementation of the paper **Task-Aware Active Learning for Endoscopic Image Analysis**.\
arxiv: https://arxiv.org/pdf/2204.03440.pdf

# Abstract
Semantic segmentation of polyps and depth estimation are two important research problems in endoscopic image analysis. One of the main obstacles to conduct research on these research problems is lack of annotated data. Endoscopic annotations necessitate the specialist knowledge of expert endoscopists and due to this, it can be difficult to organise, expensive and time consuming. To address this problem, we investigate an active learning paradigm to reduce the number of training examples by selecting the most discriminative and diverse unlabelled examples for the task taken into consideration. Most of the existing active learning pipelines are task-agnostic in nature and are often suboptimal to the end task. In this paper, we propose a novel task-aware active learning pipeline and applied for two important tasks in endoscopic image analysis: semantic segmentation and depth estimation. We compared our method with the competitive baselines. From the experimental results, we observe a substantial improvement over the compared baselines.

# Installation

Clone the repo:
```bash
    git clone https://github.com/thetna/endo-active-learn.git
```
All the required python packages can be installed with:
```bash
    cd endo-active-learn
    pip install -r requirements.txt
```

# Training

For training on depth dataset, use the following command
```bash
    python main.py --n_epochs 100 --output_path your_result_path --method al_method --num_gen_steps 2
```

Replace *your_result_path* to the path you want to store checkpoints and intermediate results in.
Replace *al_method* with one of the following options:
- CoreSet
- CoreSetPCA
- Random
- VAAL

For training dataset on depth estimation, dataset from http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/ was used. Following commands can be executed in sequence to get datasets for depth estimation
```bash
wget http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/Data/T1.zip
wget http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/Data/T2.zip
wget http://cmic.cs.ucl.ac.uk/ColonoscopyDepth/Data/T3.zip
mkdir T1 | mkdir T2 | mkdir T3
unzip T1.zip -d T1/ 
unzip T2.zip -d T2/ 
unzip T3.zip -d T3/
python scripts.pys
```
For training on segmentation dataset, first download kvasir-seg.zip file from the link https://datasets.simula.no/kvasir-seg/ and extract it to your preferred location. Then, use the following command
```bash
    python train_seg.py --n_epochs 100 --train_dir tdr --output_path your_result_path --method al_method 
```
Replace *your_result_path* to the path you want to store checkpoints and intermediate results in. Replace *tdr* with the path to kvasir-seg dataset path. 
Replace *al_method* with one of the above options. We have also tested uncertainty based methods in this dataset. For that, you can choose one of the following options:

- UncertainwithCoreset
- UncertainwithPCA

When training with method that requires PCA, PCA of datasets needed to be computed. Use following command for PCA calculation
```bash
    python pca.py --task your_task
```
Replace *your_task* with one of the following options:

- depth
- segs

For visualization of losses and metrics, [wandb](https://wandb.ai/) was used. So wandb should be configured in your machine before training the model.
# Citation
```bash
@article{thapa2022task,
  title={Task-Aware Active Learning for Endoscopic Image Analysis},
  author={Thapa, Shrawan Kumar and Poudel, Pranav and Bhattarai, Binod and Stoyanov, Danail},
  journal={arXiv preprint arXiv:2204.03440},
  year={2022}
}
```

