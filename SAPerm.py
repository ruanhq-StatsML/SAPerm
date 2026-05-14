#Sentiment Analysis Application on the world embedding

from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
import transformers
#import gensim
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
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import xgboost
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion 
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split


"""
Extract the text embedding from the last hidden output layer here and conduct 
RFPerm and GBDTPerm on the extracted contextual embeddings.

"""

class ExtractEmbedding(nn.Module):
    def __init__(self, fine_tuning, model, max_length = 128):
        self.fine_tuning = fine_tuning
        self.model = model
        self.max_length = max_length
    def extract_embedding(self, type = "Bert"):
        self.model.eval()
        all_embeddings = []
        if type == "Bert":
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_text = texts[i:(i + batch_size)]
                    encoded = tokenizer(
                        batch_text, padding = True,
                        truncation = True,
                        max_length = self.max_length,
                        return_tensor = 'pt'
                    )
                    encoded = {k: v.to(device) for k, v in encoded.items()}
                    outputs = self.model(**encoded)  
                    embeddings = self.mean_pooling(embeddings.cpu.numpy())
        return np.vstack(embeddings)  
    #Extract bert embeddings:
    def bert_base_embedding(self, X, output_dim):
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        model = AutoModel.from_pretrained('bert-base-uncased')
        inputs = tokenizer(X, return_tensors = 'pt', padding = True, truncation = True)
        with torch.no_grad():
            model_output = model(**inputs)
        attention_mask = inputs['mask']
        output_seq = model_output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(output_seq.size()).float()
        sum_embeddings = (output_seq * mask).sum(1)
        sum_mask = torch.clamp(mask.sum(1), min = 1e-7)
        mean_pooled = (sum_embeddings/sum_mask).cpu()
        return pd.Series(mean_pooled.numpy().reshape(-1)[:output_dim])
    #Extract GPT2 embeddings:
    def gpt2_tokenizer(self, X, output_dim):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        encoded_caption = tokenizer(X, return_tensors='pt')
        input_ids = encoded_caption['input_ids']
        with torch.no_grad():
            outputs = model(input_ids)
            word_embeddings = outputs.last_hidden_state
        return pd.Series(word_embeddings.sum(1).reshape(-1)[:n_dim])
    def _sim_seed(self, i: int, j: int) -> int:
        return int((9026 + i * 100003 + j * 10007)%(2**31 - 1))
    #Extract the embeddings:
    def permTest(self, df1, df2):













