#fine_tuning data extraction:
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
from fine_tuning_embedding import * 
from fine_tuning_gpt2_embedding import * 

'''
Function for dimensionality reduction with different methods:
ICA, PCA, TSNE and UMAP
'''
def decompose(df,
  method = 'PCA',
  n_components_PCA = 100, n_TSNE = 20,
  n_components_ICA = 20, n_neighbors_umap = 20,
  n_components_umap = 20,
   ABTT = 0, random_state = 2026):
    if method == 'PCA': 
        model = PCA(n_components = n_components_PCA)
        df_emb = model.fit_transform(df)
        if ABTT > 0:
            df_emb = df_emb[:, ABTT:]
    elif method == 'TSNE':
        model = TSNE(n_components = n_components_TSNE,
        perplexity = 10.0, random_state = random_state)
        df_emb = model.fit_transform(df)
    elif method == 'ICA':
        model = FastICA(n_components = n_components_ICA, 
        random_state = random_state, whiten = 'unit-variance')
        df_emb = model.fit_transform(df)
    elif method == 'UMAP':
        model = umap.UMAP(n_neighbors = n_neighbors_umap,
        min_dist = 0.25, n_components = n_compon)
        df_emb = model.fit_transform(df)
    return df, model



'''
Wrapper for the bert fine-tuning procedure:
'''
def fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 6, num_train_epochs_lora = 3,
    max_length = 128, lora_r = 16, lora_alpha = 32, lora_dropout = 0.1,
    target_modules = ['query', 'value'],
    batch_size = 32, output_csv_emb = 'X_emb_semEval2022_bertFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_bertFineTuningLora.csv', column_to_join = None):
    res = pretrain_bert_mlm_lora(
        df, text_col = text_col, base_model_id = model_name,
        output_dir = output_dir, num_train_epochs = num_train_epochs_mlm,
        max_length = max_length, lora_r = lora_r,
        lora_alpha = lora_alpha, lora_dropout = lora_dropout,
        target_modules = ['query', 'value']
    )
    X_emb_lora = bert_mean_pool_embeddings_with_lora(
        df[text_col], base_model_id = model_name, lora_dir = lora_dir,
        batch_size = batch_size, max_length = max_length
    )
    result = pretrain_bert_mlm(df, text_col = text_col,
             output_dir="./bert_mlm_adapted",
             num_train_epochs=num_train_epochs_mlm, max_length=max_length)
    X_emb = bert_mean_pool_embeddings(df[text_col],
             model_name = model_name)
    if column_to_join is None:
        pd.DataFrame(X_emb).to_csv(output_csv_emb)
        pd.DataFrame(X_emb_lora).to_csv(output_csv_lora)
    else:
        X_emb = pd.DataFrame(X_emb)
        for col in column_to_join:
            X_emb[col] = df[col]
        X_emb.to_csv(output_csv_emb)
        X_emb_lora = pd.DataFrame(X_emb_lora)
        for col in column_to_join:
            X_emb_lora[column_to_join] = df[column_to_join]
        X_emb_lora.to_csv(output_csv_lora)
        

'''
Wrapper for the GPT fine-tuning procedure:
'''
def fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 6, num_train_epochs_lora = 3,
    max_length = 128, lora_r = 16, lora_alpha = 32, lora_dropout = 0.1,
    target_modules = ['query', 'value'],
    batch_size = 32, output_csv_emb = 'X_emb_semEval2022_bertFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_bertFineTuningLora.csv', column_to_join = None):
    res = pretrain_bert_mlm_lora(
        df, text_col = text_col, base_model_id = model_name,
        output_dir = output_dir, num_train_epochs = num_train_epochs_mlm,
        max_length = max_length, lora_r = lora_r,
        lora_alpha = lora_alpha, lora_dropout = lora_dropout,
        target_modules = ['query', 'value']
    )
    X_emb_lora = gpt_embedding_with_lora(
        df[text_col], base_model_id = model_name,
        batch_size = batch_size, max_length = max_lengths
    )
    result = pretrain_bert_mlm(df, text_col = text_col,
             output_dir=output_dir,
             num_train_epochs=num_train_epochs_mlm, max_length=max_length)
    X_emb = gpt2_mean_pool_embeddings(df[text_col],
        model_name = model_name)
    if column_to_join is None:
        pd.DataFrame(X_emb).to_csv(output_csv_emb)
        pd.DataFrame(X_emb_lora).to_csv(output_csv_lora)
    else:
        X_emb = pd.DataFrame(X_emb)
        for col in column_to_join:
            X_emb[col] = df[col]
        X_emb.to_csv(output_csv_emb)
        X_emb_lora = pd.DataFrame(X_emb_lora)
        for col in column_to_join:
            X_emb_lora[column_to_join] = df[column_to_join]
        X_emb_lora.to_csv(output_csv_lora)
        




"""
SemEval2022
"""
df_train = pd.read_csv("train.En.csv")
df_test = pd.read_csv("test.En.csv")
df = pd.concat([df_train, df_test], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_semEval2022_bertFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_bertFineTuningLora.csv',
    column_to_join = ['label'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_semEval2022_gptFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_gptFineTuningLora.csv',
    column_to_join = ['label'])


"""
Sarcasm data: 
"""
df1 = pd.read_csv("HYP-sarc-notsarc.csv")
df2 = pd.read_csv("RQ-sarc-notsarc.csv")
df3 = pd.read_csv("GEN-sarc-notsarc.csv")
df = pd.concat([df1, df2, df3], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_sarc2_bertFineTuning.csv',
    output_csv_lora = 'X_emb_sarc2_bertFineTuningLora.csv',
    column_to_join = ['label'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_sarc2_gptFineTuning.csv',
    output_csv_lora = 'X_emb_sarc2_gptFineTuningLora.csv',
    column_to_join = ['label'])


"""
Equity Evaluation Data: Join the label(response) column back.
"""
df = pd.read_csv("semeval2018_taskA.txt", sep='\t')
df.columns = ["index", "label", "text"]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_sarc2_bertFineTuning.csv',
    output_csv_lora = 'X_emb_sarc2_bertFineTuningLora.csv',
    column_to_join = ['label'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_sarc2_gptFineTuning.csv',
    output_csv_lora = 'X_emb_sarc2_gptFineTuningLora.csv',
    column_to_join = ['label'])


"""
Gender Bias Data: Join the sentiment(response) column back.
"""
df = pd.read_csv("explainability_gender_bias_data.csv")
df.columns = ["index", "text", "sentiment"]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_explainability_gender_bias_data_bertFineTuning.csv',
    output_csv_lora = 'X_emb_explainability_gender_bias_data_bertFineTuningLora.csv',
    column_to_join = ['sentiment'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_explainability_gender_bias_data_gptFineTuning.csv',
    output_csv_lora = 'X_emb_explainability_gender_bias_data_gptFineTuningLora.csv',
    column_to_join = ['sentiment'])


"""
Bias Finder Data: Join the label(response) column back.
"""
train_bias = pd.read_csv("dataset/train_bias2.csv", sep='\t', header=None)
train_bias.columns = ["sentiment", "text"]
test_bias = pd.read_csv("dataset/test_bias2.csv", sep='\t', header=None)
test_bias.columns = ["sentiment", "text"]
df = pd.concat([train_bias, test_bias], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_biasfinder_bertFineTuning.csv',
    output_csv_lora = 'X_emb_biasfinder_bertFineTuningLora.csv',
    column_to_join = ['sentiment'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_biasfinder_gptFineTuning.csv',
    output_csv_lora = 'X_emb_biasfinder_gptFineTuningLora.csv',
    column_to_join = ['label'])


"""
Twitter Data: Join the label(response) column back.
"""
df = pd.read_csv("twitter.csv")
df.columns = ["id", "label", "text"]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_twitter_bertFineTuning.csv',
    output_csv_lora = 'X_emb_twitter_bertFineTuningLora.csv',
    column_to_join = 'label')
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_twitter_gptFineTuning.csv',
    output_csv_lora = 'X_emb_twitter_gptFineTuningLora.csv',
    column_to_join = 'label')


"""
Amazon Subscription Magazine Dataset, join the rating(response) column back.
"""
df = pd.read_csv("magazineSubscription_df.csv")
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_magazine_Amazon_bertFineTuning.csv',
    output_csv_lora = 'X_emb_magazine_Amazon_bertFineTuningLora.csv',
    column_to_join = ['rating'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_magazine_Amazon_gptFineTuning.csv',
    output_csv_lora = 'X_emb_magazine_Amazon_gptFineTuningLora.csv',
    column_to_join = ['rating'])

"""
Amazon Gift Card Dataset, join the rating(response) column back.
"""
with open('Gift_Cards.jsonl') as f:
    lines = f.read().splitlines()
line_dicts = [json.loads(line) for line in lines]
df = pd.DataFrame(line_dicts)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_metaGiftCards_Amazon_bertFineTuning.csv',
    output_csv_lora = 'X_emb_metaGiftCards_Amazon_bertFineTuningLora.csv',
    column_to_join = ['rating'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_metaGiftCards_Amazon_gptFineTuning.csv',
    output_csv_lora = 'X_emb_metaGiftCards_Amazon_gptFineTuningLora.csv',
    column_to_join = ['rating'])


"""
Amazon Gift Card Dataset, join the rating(response) column back.
"""
df = pd.read_csv("Equity-Evaluation-Corpus.csv")
df["text"] = df["Sentence"]

fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_equityevaluation_bertFineTuning.csv',
    output_csv_lora = 'X_emb_equityevaluation_bertFineTuningLora.csv',
    column_to_join = ['gender', 'race'])
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_equityevaluation_gptFineTuning.csv',
    output_csv_lora = 'X_emb_equityevaluation_gptFineTuningLora.csv',
    column_to_join = ['gender', 'race'])















