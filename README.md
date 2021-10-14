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

Download the trained LaserTagger model from [GDrive](https://drive.google.com/drive/folders/11q4pXX_MPB8P-XNdDfznq9KhnhcMZqol?usp=sharing). Put the contens of this zip folder into `src/laserTagger/models` folder.

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
@inproceedings{upadhyay2021case,
  title={A Case-Based Approach to Data-to-Text Generation},
  author={Upadhyay, Ashish and Massie, Stewart and Singh, Ritwik Kumar and Gupta, Garima and Ojha, Muneendra},
  booktitle={International Conference on Case-Based Reasoning},
  pages={232--247},
  year={2021},
  organization={Springer}
}
```
