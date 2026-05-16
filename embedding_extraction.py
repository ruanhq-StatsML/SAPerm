#Extracting all of the embeddings:
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import transformers
import gensim
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import BartForConditionalGeneration, BartTokenizer, BartTokenizerFast, BartModel
from transformers import FlaxPreTrainedModel, FlaxBartForCausalLM, AutoModelForCausalLM, FlaxGPT2Model
from nltk.tokenize import word_tokenize
import statistics
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split

#Extracting the embeddings:

def gpt2tokenizer(X, n_dim = 200):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    encoded_caption = tokenizer(X, return_tensors='pt')
    input_ids = encoded_caption['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
        word_embeddings = outputs.last_hidden_state
    return pd.Series(word_embeddings.sum(1).reshape(-1)[:n_dim])

def bert_base_embedding(X, n_dim = 30):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    inputs = tokenizer(X, return_tensors = 'pt', padding = True, truncation = True)
    with torch.no_grad():
        model_output = model(**inputs)
    attention_mask = inputs['attention_mask']
    output_seq = model_output.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(output_seq.size()).float()
    sum_embeddings = (output_seq * mask).sum(1)
    sum_mask = torch.clamp(mask.sum(1), min = 1e-7)
    mean_pooled = (sum_embeddings/sum_mask).cpu()
    return pd.Series(mean_pooled.numpy().reshape(-1)[:n_dim])

def embedding_extraction(df, text_col, column_to_join, embedding_type = 'bert',
                              output_csv = 'Equity_Evaluation_Corpus_bert_100PC_ABTT10.csv',
                              PCA = True, n_pc = 100, ABTT = 10, n_dim = 768):
    """
    Extracting the Embeddings - Bert/GPT with Optional PCA embedding procedure
    """
    if embedding_type == 'bert':
        df_emb = df[text_col].apply(lambda X: bert_base_embedding(X, n_dim = n_dim))
    elif embedding_type == 'gpt2':
        df_emb = df[text_col].apply(lambda X: gpt2tokenizer(X, n_dim = n_dim))
    if PCA:
        pca_obj = PCA(n_components = n_pc)
        df_emb = pd.DataFrame(pca_obj.fit_transform(df_emb))
        if ABTT > 0:
            df_emb = df_emb[:, ABTT:]
    output_df = pd.concat([df_emb, df[column_to_join]], axis = 1)
    output_df.to_csv(output_csv)











