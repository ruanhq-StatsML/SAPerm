#Before running it, update the torchao versions: -- !pip install --upgrade torchao
from __future__ import annotations
import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed
)
from concurrent.futures import ProcesPoolExecutor, as_completed
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from FT_data_extraction_utils import *

'''
Function for dimensionality reduction with different methods:
ICA, PCA, TSNE and UMAP
'''





###################################
df = pd.read_csv("Datasets/Equity-Evaluation-Corpus.csv")
df["text"] = df["Sentence"]

#subsampling for illustration purpose, for reproducibility, run:
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_equityevaluation_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_equityevaluation_bertFineTuningLora.csv',
    column_to_join = ['Gender', 'Race'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_equityevaluation_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_equityevaluation_gptFineTuningLora.csv',
    column_to_join = ['Gender', 'Race'])


"""
SemEval2022
"""
#For the subsampling procedure:
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
df_train = pd.read_csv("train.En.csv")
df_test = pd.read_csv("test.En.csv")
df = pd.concat([df_train, df_test], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_semEval2022_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_semEval2022_bertFineTuningLora.csv',
    column_to_join = ['label'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_semEval2022_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_semEval2022_gptFineTuningLora.csv',
    column_to_join = ['label'])


"""
Sarcasm data: 
"""
#For the subsampling procedure:
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
df1 = pd.read_csv("HYP-sarc-notsarc.csv")
df2 = pd.read_csv("RQ-sarc-notsarc.csv")
df3 = pd.read_csv("GEN-sarc-notsarc.csv")
df = pd.concat([df1, df2, df3], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_sarc2_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_sarc2_bertFineTuningLora.csv',
    column_to_join = ['label'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_sarc2_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_sarc2_gptFineTuningLora.csv',
    column_to_join = ['label'])


"""
Equity Evaluation Data: Join the label(response) column back.
"""
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
df = pd.read_csv("semeval2018_taskA.txt", sep='\t')
df.columns = ["index", "label", "text"]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_equitEval_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_equitEval_bertFineTuningLora.csv',
    column_to_join = ['label'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_equitEval_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_equitEval_gptFineTuningLora.csv',
    column_to_join = ['label'])


"""
Gender Bias Data: Join the sentiment(response) column back.
"""
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
df = pd.read_csv("explainability_gender_bias_data.csv")
df.columns = ["index", "text", "sentiment"]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_explainability_gender_bias_data_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_explainability_gender_bias_data_bertFineTuningLora.csv',
    column_to_join = ['sentiment'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_explainability_gender_bias_data_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_explainability_gender_bias_data_gptFineTuningLora.csv',
    column_to_join = ['sentiment'])


"""
Bias Finder Data: Join the label(response) column back.
"""
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
train_bias = pd.read_csv("dataset/train_bias2.csv", sep='\t', header=None)
train_bias.columns = ["sentiment", "text"]
test_bias = pd.read_csv("dataset/test_bias2.csv", sep='\t', header=None)
test_bias.columns = ["sentiment", "text"]
df = pd.concat([train_bias, test_bias], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_biasfinder_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_biasfinder_bertFineTuningLora.csv',
    column_to_join = ['sentiment'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_biasfinder_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_biasfinder_gptFineTuningLora.csv',
    column_to_join = ['sentiment'])



"""
Twitter Data: Join the label(response) column back.
"""
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
df = pd.read_csv("twitter.csv")
df.columns = ["id", "label", "text"]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_twitter_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_twitter_bertFineTuningLora.csv',
    column_to_join = ['sentiment'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_twitter_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_twitter_gptFineTuningLora.csv',
    column_to_join = ['sentiment'])


"""
Amazon Subscription Magazine Dataset, join the rating(response) column back.
"""
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
df = pd.read_csv("magazineSubscription_df.csv")
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_magazine_Amazon_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_magazine_Amazon_bertFineTuningLora.csv',
    column_to_join = ['rating'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_magazine_Amazon_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_magazine_Amazon_gptFineTuningLora.csv',
    column_to_join = ['rating'])



"""
Amazon Gift Card Dataset, join the rating(response) column back.
"""
#np.random.seed(2026)
#df = df.iloc[np.random.choice(np.arange(df.shape[0], 5000, replace = False))]
with open('Gift_Cards.jsonl') as f:
    lines = f.read().splitlines()
line_dicts = [json.loads(line) for line in lines]
df = pd.DataFrame(line_dicts)

fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_metaGiftCards_Amazon_bertFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_metaGiftCards_Amazon_bertFineTuningLora.csv',
    column_to_join = ['rating'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 7, num_train_epochs_lora = 5,
    output_csv_emb = 'Resulted_data/X_emb_metaGiftCards_Amazon_gptFineTuning.csv',
    output_csv_lora = 'Resulted_data/X_emb_metaGiftCards_Amazon_gptFineTuningLora.csv',
    column_to_join = ['rating'])















