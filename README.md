# Data2Text-CBR
Data-to-Text Generation with Case-Based Reasoning

## Requirements

Run:
```bash
pip install -r requirements.txt
```


## How to Run

### Download the GPT2 finetuned model

Download the fine-tuned GPT2 model from [GDrive](https://drive.google.com/drive/folders/11q4pXX_MPB8P-XNdDfznq9KhnhcMZqol?usp=sharing).
It's a zip folder, unzip the files into a `gpt2-finetuned` folder in root directory.

### Download the LaserTagger model

Download the trained LaserTagger model from [GDrive](https://drive.google.com/file/d/1uZI-ozhOj2KwzDjZDbgTro2JplDLGSXA/view?usp=sharing). Put the contens of this zip folder into `src/laserTagger/models` folder.

You'll also need to download a pretrained BERT model from the [official repository](https://github.com/google-research/bert#pre-trained-models).
You need to download the 12-layer ''BERT-Base, Cased'' model. Put the contents inside `src/laserTagger/bert` folder.

Note: there might be some issues with the TensorFlow version used in LaserTagger. You might need to run it in a virtua-environment then. Anyhow, even without LaserTagger generation can be done and there won't be any noticable difference in the metric scores.

### Run

```bash
sh final.sh
```

1. Create clusters
2. Train Feature Weighting
3. Train important player classifier
4. Create Case-Base
5. Do generation
6. Apply LaserTagger


## Cite

```latex
@InProceedings{10.1007/978-3-030-86957-1_16,
author="Upadhyay, Ashish
and Massie, Stewart
and Singh, Ritwik Kumar
and Gupta, Garima
and Ojha, Muneendra",
editor="S{\'a}nchez-Ruiz, Antonio A.
and Floyd, Michael W.",
title="A Case-Based Approach to Data-to-Text Generation",
booktitle="Case-Based Reasoning Research and Development",
year="2021",
publisher="Springer International Publishing",
address="Cham",
pages="232--247",
abstract="Traditional Data-to-Text Generation (D2T) systems utilise carefully crafted domain specific rules and templates to generate high quality accurate texts. More recent approaches use neural systems to learn domain rules from the training data to produce very fluent and diverse texts. However, there is a trade-off with rule-based systems producing accurate text but that may lack variation, while learning-based systems produce more diverse texts but often with poorer accuracy. In this paper, we propose a Case-Based approach for D2T that mitigates the impact of this trade-off by dynamically selecting templates from the training corpora. In our approach we develop a novel case-alignment based, feature weighing method that is used to build an effective similarity measure. Extensive experimentation is performed on a sports domain dataset. Through Extractive Evaluation metrics, we demonstrate the benefit of the CBR system over a rule-based baseline and a neural benchmark.",
isbn="978-3-030-86957-1"
}

```
