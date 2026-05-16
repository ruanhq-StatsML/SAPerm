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

def extract_embedding_df(input_csv, )