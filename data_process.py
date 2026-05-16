#Process data:
#Amazon Review Dataset - Json to dataframe:
import json
import pandas as pd
from typing import Any, Dict, Optional
import numpy as np
from sklearn.decomposition import PCA, UMAP, TSNE, LDA, KernelPCA, FastICA


'''
This implements eigen shift on the contextual word embedding, different types of Shifts injected:
'''
def eigen_shift(df, n_components_PCA = 25, random_state = 2026):
    if standardize:
        scaler = StandardScaler()
        X_user = scaler.fit_transform(X)
    else:
        scaler = None
        X_user = X - X.mean(axis = 0, keepdims = True)
    pca = PCA(n_components = n_components_PCA, random_state = random_state)
    pca.fit(X_user)
    return PCA, X_user, scaler

def apply_eigen_shift(X, direction, gamma):
    if standardize and scaler is not None:
        X_std = scaler.transform(X)
        X_new_std = X_std + gamma * direction.reshape(1, -1)
        X_new = scaler.inverse_transform(X_new_std)
    else:
        X_new = X + gamma * direction.reshape(1, -1)
    return X_new


"""
This script implements all of the translational & procedures for the - including the following functions:
- back_translation:                 Translate the English to another English then translate back to English.
- Injection:                        Syntax, random character injection
- Random Negative Words Injection:  Insertion of the random negative words
- Dimensionality Reduction:         For the Contextual Text Embedding
- Rewrite:                          Rephrase via different styles of the text(casual/professional) via calling the OpenAI API
"""

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


def amazon_review_df(json_file, out_csv):
	lines = []
    with open(json_file) as f:
        line = f.read().splitlines()
    line_dicts = [json.loads(line) for line in lines]
    df = pd.DataFrame(line_dicts)
    df.to_csv(out_csv)


model_en_de = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-en-de')
model_de_en = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-de-en')
tokenizer_en = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-de')
tokenizer_de = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-de-en')
FROM_MODEL = "facebook/wmt19-en-de"
TO_MODEL = "facebook/wmt19-de-en"


#Back Translation:
def back_translation(tokenizer, model, texts, batch_size = 16, max_length = 128):
	out = []
	model.to(device)
    for i in range(0, len(texts), batch_size):
        batch_text = [str(x).strip() if str(x).strip() else '.'
            for x in texts[i:(i + batch_size)]]
        #Padding is required here:
        encoders = tokenizer(
            batch, return_tensors = 'pt', padding = True,
            truncation = True, max_length = max_length
        ).to(device)#ensure the same length.
        gen_kw = {'max_length': max_length}
        gen_kw['num_beams'] = num_beams
        gen_kw['early_stopping'] = True
        with torch.inference_mode():
            output = model.generate(**encoders, max_length = max_length)
        out.extend(tokenizer.batch_decode(output, skip_special_tokens = True))
    return out

def back_translate_texts(
    texts, tokenizer1,
    model1, tokenizer2,
    model2, device = device,
    batch_size = 16, max_length = 128,
    num_beams = 2):
    de = back_translation(tokenizer1, model1,
        texts = texts, device = device, batch_size = batch_size,
        max_length = max_length)
    output = back_translation(tokenizer2, model2,
        texts = de, device = device, batch_size = batch_size,
        max_length = max_length)
    return output


'''
Text Injection: Rephrase & Adding negative words.
'''
ascii_special = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
whitespace_control = '\n\t\r'
unicode_variants = '“”‘’–—…•·°±×÷²³'
transformer_tokens = '[UNK][PAD][CLS][SEP][MASK]<s></s>'
#Asking the 
SPECIAL_CHARACTERS = ascii_special + whitespace_control + unicode_variants + transformer_tokens
class TextInjection:
    def __init__(self, levels, client, 
        model = 'gpt-4o-mini',
        negative_word_dict):
        self.client = OpenAI(api_key = API_KEY)
        self.model = model
        self.negative_word_dict = negative_word_dict
    def get_variant(self, text, ):
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
            {'role': 'system', 'content': self.levels[level_key]},
            {'role': 'user', 'content': str(text)},
            ],
            temperature = 0.7 if 'L3' in level_key else 0.25
        )
        return response.choices[0].message.content.strip()
    def add_negative_word(self, text, k):
        t = str(text)
        for _ in range(k):
            t_list = list(t)
            idx = random.randint(1, len(t_list))
            rand_neg_word = negative_word_dict[random.randint(1, len(negative_word_dict))]
            t_list = t[:idx] + rand_neg_word + t[(idx + 1):]
            t = ''.join(t_list)
        t = t.rstrip('.') + ('...' if random.random() < 0.5 else '!')
        return t
    #randomly change the order of a couple of words with lexical noise:
    def add_lexical_noise(self, text, k):
    	t = str(text)
    	if len(t) > 5:
    	    for _ in range(k):
    	    	t_list = list(t)
    	        idx = random.randint(1, len(t_list) - 1)
    	        t_list[idx], t_list[idx - 1] = t_list[idx], t_list[idx - 1]
    	        t = ''.join(t_list)
    	t = t.rstrip('.') + ('...' if random.random() < 0.5 else '!')
    	return t
    #randomly inject some of the special characters, not perturbing the predictive relationship and the semantic meaning:
    def add_special_char(self, text, k, special_char_list):
        special_char_list = SPECIAL_CHARACTERS
        t = str(text)
        if len(t) > 5:
            #adding some of the random noise in between the words.
            for _ in range(round(k)):
                idx = random.randint(1, len(t) - 1)
                t = t[:idx] + SPECIAL_CHARACTERS[np.random.randint(1, len(SPECIAL_CHARACTERS) - 1)] + t[(idx + 1):]
        return t
    #add syntax noise:
    def add_syntax_noise(self, text, del_prop, rep_prob, k = 3):
        t = str(text)
        for j in range(k):
        	random_n = np.random.random()
            if np.random.random() > 0.5:
            	t = ''.join([char.capitalize() for char in t])
            if len(t) > 6:
                idx = random.randint(1, len(t) - 1)
                t_list = list(t)
                t_list[idx], t_list[idx - 1] = t_list[idx - 1], t_list[idx]
                t = ''.join(t_list)
            if np.random.random() < float(del_prop):
                idx = np.random.randint(0, max(len(t) - 1, 0))
                t = t[:idx] + t[(idx + 1):]
        t = t.rstrip('.') + ('...' if random.random() < 0.5 else '!')
        return t
    def rewrite(self, text, level = 'rephrase', style = 'formal'):
        prompts = {
        'syntax': (
            f"Rewrite this text to be more {style} and concise."
             "Keep the original meaning and high-level intuition."
            ),
        'paraphrase':(
            'Paraphrase the following text using completely different vocabulary while strictly preserving the meaning.'
            ),
        }
        response = self.client.chat.completions.create(
            model = self.model,
            messages = [
            {'role': 'system',
             'content': (
                'You are a precise text transformation engine. Output only the trasnformed text.'
                )},
             {'role': 'user', 'content': f"{prompts.get(level, level)}: {text}"},
            ],
            temperature = 0.8 if level == 'paraphrase' else 0.25
        )
        return response.choices[0].message.content.strip()









