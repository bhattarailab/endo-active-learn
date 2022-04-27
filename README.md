# Task-Aware Active Learning for Endoscopic Image Analysis

Official PyTorch implementation of the paper **Task-Aware Active Learning for Endoscopic Image Analysis**

# Abstract
Semantic segmentation of polyps and depth estimation are two important research problems in endoscopic image analysis. One of the main obstacles to conduct research on these research problems is lack of annotated data. Endoscopic annotations necessitate the specialist knowledge of expert endoscopists and due to this, it can be difficult to organise, expensive and time consuming. To address this problem, we investigate an active learning paradigm to reduce the number of training examples by selecting the most discriminative and diverse unlabelled examples for the task taken into consideration. Most of the existing active learning pipelines are task-agnostic in nature and are often suboptimal to the end task. In this paper, we propose a novel task-aware active learning pipeline and applied for two important tasks in endoscopic image analysis: semantic segmentation and depth estimation. We compared our method with the competitive baselines. From the experimental results, we observe a substantial improvement over the compared baselines.

# Training

For training on depth dataset:
```bash
    python main.py --n_epochs 100 --output_path *your_result_path* --method *al_method* --num_gen_steps 2
```
Replace *your_result_path* to the path you want to store checkpoints and intermediate results in.
Replace *al_method* with one of the following options:
- CoreSet
- CoreSetPCA
- Random
- VAAL


