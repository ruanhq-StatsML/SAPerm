#FT for the domain shift and sarcasm detection use cases:
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
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from fine_tuning_bert_embedding import *




