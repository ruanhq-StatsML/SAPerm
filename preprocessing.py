#data preprocessing:
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from transformers import AutoTokenizer


def data_process(csv_df, ctr_col, trt_col, response_col):
    df = pd.read_csv(csv_df)
    data_list = []
    for idx, row in df.iterrows():
        data_list.append({
            'text': str(row[ctr_col]),
            'T': 0.0,
            'Y': float(row[response_col])
        })
        data_list.append({
            'text': str(row[trt_col]),
            'T': 1.0,
            'Y': float(row[response_col])
        })
    df_output = pd.DataFrame(data_list)
    return df_output    

def orthogonal_r_loss(y_true, t_true, m_hat, e_hat, tau_hat, ortho_reg = 0.25):
    y_res = y_true - m_hat
    t_res = t_true - e_hat
    standard_r_loss = torch.mean((y_res - t_res * tau_hat) ** 2)
    #orthogonalization term:
    epsilon = y_res - t_res * tau_hat
    epsilon_centered = epsilon - torch.mean(epsilon)
    t_res_centered = t_res - torch.mean(t_res)
    ortho_penaty = torch.mean((epsilon_centered * t_res_centered) ** 2)
    ortho_term = ortho_reg * ortho_penaty
    total_loss = ortho_term + standard_r_loss
    return total_loss, ortho_term, standard_r_loss

class SarcasmDataset(Dataset):
    def __init__(self, texts, T, Y, tokenizer, max_len = 256):
        self.texts = texts
        self.T = T
        self.Y = Y
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text, add_special_tokens = True,
            max_length = self.max_len, padding = 'max_length',
            truncation = True, return_tensor = 'pt'
        )
        return {
        'input_ids': encoding['input_ids'].flatten(),
        'attention_mask': encoding['attention_mask'].flatten(),
        'T': torch.tensor(self.T[idx], dtype = torch.float32),
        'Y': torch.tensor(self.Y[idx], dtype = torch.float32)
        }

    

class OrthogonalRoBERTa(nn.Module):
    def __init__(self, model_name = 'bert-base-uncased', config = None):
        super(OrthogonalRoBERTa, self).__init__()
        self.encoder = RobertaModel.from_pretrained(model_name)
        d_model = self.encoder.config.hidden_size
        h_dims = config.get('hidden_dims') if config else [128, 32]
        self.heads = nn.ModuleDict({
            'm_head': self.MLP(d_model, h_dims, out_dim = 1), #m(X)
            'e_head': self.MLP(d_model, h_dims, out_dim = 1),
            'tau_head': self.MLP(d_model, h_dims, out_dim = 1)
        })
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def MLP(self, input_dim, hidden_dims_list, out_dim):
        layers = []
        curr_dim = input_dim
        for h_dim in hidden_dims_list:
            layers.append(nn.Linear(curr_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            curr_dim = h_dim
        layers.append(nn.Linear(curr_dim, out_dim))
        return nn.Sequential(*layers)
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids = input_ids, attention_mask = attention_mask)
        z = outputs.last_hidden_state[:, 0, :]
        m_val = self.heads['m_head'](z).squeeze(-1)
        e_prob = torch.sigmoid(self.heads['e_head'](z)).squeeze(-1)
        tau_val = self.heads['tau_head'](z).squeeze(-1)
        return m_val, e_prob, tau_val
    def initialize_weights(self):
        for name, module in self.heads.named_modules():
            nn.init.kaiming_normal_(module.weight, mode = 'fan_out',
                nonlinearity = 'relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        #####
        last_linear = self.head['tau_head'][-1]
        nn.init.normal_(last_linear.weight, mean = 0.0, std = 0.01)    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = data_process('train.En.csv', ctr_col = 'tweet', trt_col = 'rephrase',
    response_col = 'sarcasm').sample(1000, random_state = 20).reset_index(drop = True)
tokenizer = RobertaTokenizer.from_pretrained('bart-base-uncased')
dataset = SarcasmDataset(df['text'].values, df['T'].values, df['Y'].values, tokenizer)
dataloader = DataLoader(dataset, batch_size = 16, shuffle = True)
model = OrthogonalRoBERTa(model_name = 'bart-base-uncased')
optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-5)
model.train()
for epoch in range(10):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        # Stack the list of tensors into a single tensor and remove the singleton dimension
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        y_true = batch['Y'].to(device)
        t_true = batch['T'].to(device)
        m_hat, e_hat, tau_hat = model.forward(
            ids, mask
        )
        loss, ortho_loss, r_loss = orthogonal_r_loss(
            ids, mask
            m_hat, e_hat, tau_hat
        )
        loss.backward()
        optimizer.step()
    print(f"Final Batch Loss: {loss.item():.4f}(R-loss: {r_loss.item():.4f}, O-loss: {ortho_loss.item():.4f})")









