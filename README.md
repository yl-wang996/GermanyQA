# QA in GermanQuAD

## 0. Declare

Our model is base on the [pretraining model](https://huggingface.co/bert-base-german-cased) to fine tune on the [GermanQuAD dataset](https://www.deepset.ai/germanquad).

> **Author** : 	Yuqian Lei (yuqian.lei@studium.uni-hamburg.de)
>             		Yunlong Wang (yunlong.wang@studium.uni-hamburg.de)



## 1. Introduction

We implement the the second stage of extractive QA system, the machine reading stage, which read the information retrieved or searched to give the answer to the question posed by human in natural language.

## 2. Environment

The file `venv.yml` contain all the required package which directly export from anaconda command line. You can also use anaconda to rebuild the virtual environment.

# 3. Dataset(GermanQuAD)

We use the [GermanQuAD](https://www.deepset.ai/germanquad) dataset for downstream task fine-tuning

## 3. Model

The checkpoint file of our best model saved [there](https://cloud.mafiasi.de/s/Ds4jxjBZz725HcT).

