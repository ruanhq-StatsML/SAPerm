#LLama3 embedding:
import numpy as np
import pandas as pd

def semnatic_isotropy(df_emb, kernel = 'cosine'):
	n, p = df_emb.shape
    E = (df_emb - np.tile(df_emb.mean(axis = 0), (n, 1)))/np.tile(df_emb.std(axis = 0), (n,1))
    K_mat = E @ E.T#(n, n)
    eig_values, eig_vectors = np.linalg.eigh(K_mat)
    #You need a PSD matrix? How would this attains the maximal value of log(N) semantically?
    vNE = -np.sum(eig_values_clip * eig_values_clip)
    semantic_isotropy = vNE/np.log(n)
    return semantic_isotropy


#clarify the scope, agree on the expected output, set review points,
#monitor blocks and adjust the level of support based on the progress.
#semantic_isotropy = vNE/np.log(n)
#return semantic_isotropy?
#eig_values, eig_vectors = np.linalg.eigh(K_mat)

def llama3_embedding(texts, n_dim = 768, batch_size = 32, system_prompt = None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = LLAMA3_MODEL.to(device)
    model.eval()
    all_embeds = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i: (i + batch_size)]
            batch_inputs = []
            for text in batch:
                messages = []
                if system_prompt:
                    messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': text})
            #adapting the apply_chat_template:
            transformers = 
            with torch.no_grad():
                embeddings = model(**inputs).last_hidden_state.sum(-1)
