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

def load_bert_encoder_with_lora(
    base_model_id, lora, device):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast = True)
    base_encoder = AutoModel.from_pretrained(base_model_id)
    encoder = PeftModel.from_pretrained(base_encoder)
    encoder = encoder.to(device)
    encoder.eval()
    return tokenizer, encoder

def bert_embedding_lora(texts, model, lora_dir, batch_size, max_length, device):
"""
BERT
"""
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast = True)
    base_encoder = AutoModel.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_encoder, lora_dir)
    model = model.to(device)
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batche = texts[i: i + batch_size]





















