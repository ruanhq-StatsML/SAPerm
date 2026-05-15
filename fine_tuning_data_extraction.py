#fine_tuning data extraction:
'''
- 
'''
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
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
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from fine_tuning_bert_embedding import * 
from fine_tuning_gpt2_embedding import * 
 


############################################################
#Extract Fine-Tuned version for Magazine Subscription Data:
############################################################
df = pd.read_csv("dataset/magazineSubscription_df.csv")
res = pretrain_bert_mlm_lora(
         df,
         text_col="text",
         base_model_id="bert-base-uncased",
         output_dir="./bert_mlm_lora_adapted",
         num_train_epochs=5,
         max_length=128,
         lora_r=16,
         lora_alpha=32,
         lora_dropout=0.1,
         target_modules=["query", "value"],
     )
X_emb_lora = bert_mean_pool_embeddings_with_lora(
         df["text"],
         base_model_id="bert-base-uncased",
         lora_dir="./bert_mlm_lora_adapted",
         batch_size=32,
         max_length=128,
     )
res = pretrain_bert_mlm(df, text_col="text",
 output_dir="./bert_mlm_adapted",
  num_train_epochs=10.0, max_length=128)
X_emb = bert_mean_pool_embeddings(df["text"],
"./bert_mlm_adapted")
pd.DataFrame(X_emb).to_csv("X_emb_magazine_Amazon_bertFineTuning.csv")
pd.DataFrame(X_emb_lora).to_csv("X_emb_magazine_Amazon_bertFineTuningLora.csv")




def fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
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
        batch_size = batch_size, max_length = max_lengths
    )
    result = pretrain_bert_mlm(df, text_col = text_col,
             output_dir="./bert_mlm_adapted",
             num_train_epochs=num_train_epochs_mlm, max_length=max_length)
    X_emb = bert_mean_pool_embeddings(df[text_col], output_dir_lora)
    if column_to_join is None:
        pd.DataFrame(X_emb).to_csv(output_csv_emb)
        pd.DataFrame(X_emb_lora).to_csv(output_csv_lora)
    else:
        X_emb = pd.DataFrame(X_emb)
        X_emb[column_to_join] = df[column_to_join]
        X_emb.to_csv(output_csv_emb)
        X_emb_lora = pd.DataFrame(X_emb_lora)
        X_emb_lora[column_to_join] = df[column_to_join]
        X_emb_lora.to_csv(output_csv_lora)

def fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
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
    X_emb_lora = gpt_embedding_with_lora(
        df[text_col], base_model_id = model_name,
        batch_size = batch_size, max_length = max_lengths
    )
    result = pretrain_bert_mlm(df, text_col = text_col,
             output_dir="./bert_mlm_adapted",
             num_train_epochs=num_train_epochs_mlm, max_length=max_length)
    if column_to_join is None:
        pd.DataFrame(X_emb).to_csv(output_csv_emb)
        pd.DataFrame(X_emb_lora).to_csv(output_csv_lora)
    else:
        X_emb = pd.DataFrame(X_emb)
        X_emb[column_to_join] = df[column_to_join]
        X_emb.to_csv(output_csv_emb)
        X_emb_lora = pd.DataFrame(X_emb_lora)
        X_emb_lora[column_to_join] = df[column_to_join]
        X_emb_lora.to_csv(output_csv_lora)
        

def fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 6, num_train_epochs_lora = 3,
    max_length = 128, lora_r = 16, lora_alpha = 32, lora_dropout = 0.1,
    target_modules = ['query', 'value'],
    batch_size = 32, output_csv_emb = 'X_emb_semEval2022_gptFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_gptFineTuningLora.csv', column_to_join = None):
    res = pretrain_gpt2_clm(
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
    result = pretrain_gpt2_clm_lora(df, text_col = text_col,
             output_dir="./bert_mlm_adapted",
             num_train_epochs=num_train_epochs_mlm, max_length=max_length)
    if column_to_join is None:
        pd.DataFrame(X_emb).to_csv(output_csv_emb)
        pd.DataFrame(X_emb_lora).to_csv(output_csv_lora)
    else:
        X_emb = pd.DataFrame(X_emb)
        X_emb[column_to_join] = df[column_to_join]
        X_emb.to_csv(output_csv_emb)
        X_emb_lora = pd.DataFrame(X_emb_lora)
        X_emb_lora[column_to_join] = df[column_to_join]
        X_emb_lora.to_csv(output_csv_lora)


pretrain_gpt2_clm
"""
SemEval2022
"""
df_train = pd.read_csv("train.En.csv")
df_test = pd.read_csv("test.En.csv")
df = pd.concat([df_train, df_test], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_semEval2022_bertFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_bertFineTuningLora.csv')
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_semEval2022_gptFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_gptFineTuningLora.csv')

"""
Sarcasm data:
"""
df1 = pd.read_csv("HYP-sarc-notsarc.csv")
df2 = pd.read_csv("RQ-sarc-notsarc.csv")
df3 = pd.read_csv("GEN-sarc-notsarc.csv")
df = pd.concat([df1, df2, df3], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_sarc2_bertFineTuning.csv',
    output_csv_lora = 'X_emb_sarc2_bertFineTuningLora.csv')
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_sarc2_gptFineTuning.csv',
    output_csv_lora = 'X_emb_sarc2_gptFineTuningLora.csv')


"""
Equity Evaluation Data:
"""
df = pd.read_csv("semeval2018_taskA.txt", sep='\t')
df.columns = ["index", "label", "text"]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_sarc2_bertFineTuning.csv',
    output_csv_lora = 'X_emb_sarc2_bertFineTuningLora.csv',
    column_to_join = 'label')
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_sarc2_gptFineTuning.csv',
    output_csv_lora = 'X_emb_sarc2_gptFineTuningLora.csv',
    column_to_join = 'label')

"""
Gender Bias Data:
"""
df = pd.read_csv("explainability_gender_bias_data.csv")
df.columns = ["index", "text", "sentiment"]
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_explainability_gender_bias_data_bertFineTuning.csv',
    output_csv_lora = 'X_emb_explainability_gender_bias_data_bertFineTuningLora.csv',
    column_to_join = 'sentiment')
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_explainability_gender_bias_data_gptFineTuning.csv',
    output_csv_lora = 'X_emb_explainability_gender_bias_data_gptFineTuningLora.csv',
    column_to_join = 'label')



"""
Bias Finder Data:
"""
train_bias = pd.read_csv("dataset/train_bias2.csv", sep='\t', header=None)
train_bias.columns = ["sentiment", "text"]
test_bias = pd.read_csv("dataset/test_bias2.csv", sep='\t', header=None)
test_bias.columns = ["sentiment", "text"]
df = pd.concat([train_bias, test_bias], axis = 0)
fine_tuning_data_wrapper_bert(df, text_col = 'text',
    output_csv_emb = 'X_emb_biasfinder_bertFineTuning.csv',
    output_csv_lora = 'X_emb_biasfinder_bertFineTuningLora.csv',
    column_to_join = 'sentiment', column_to_join = 'sentiment')
fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    output_csv_emb = 'X_emb_biasfinder_gptFineTuning.csv',
    output_csv_lora = 'X_emb_biasfinder_gptFineTuningLora.csv',
    column_to_join = 'label')



"""
Twitter Data:
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






