#Fine Tuning Data Extraction Utils:
"""
LoRA, GPT2 with the Fine-Tuning Procedure
"""
def ft_gpt2_mlm_lora(
    df: pd.DataFrame,
    text_col: str = "text",
    base_model_id: str = "gpt2",
    output_dir: str = "./gpt2_clm_lora_adapted",
    seed: int = 0,
    # Tokenization
    max_length: int = 128,
    # LoRA hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    # Training hyperparams
    num_train_epochs: float = 1.0,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    train_val_split: float = 0.98,
) -> Dict[str, Any]:
    """
    Fine-Tuning (Causal LM / next-token prediction) with LoRA adapters on df[text_col].[]
    Saves:
      - LoRA adapters to output_dir (model.save_pretrained)
      - tokenizer to output_dir
    Returns:
      dict with { "model", "tokenizer", "trainer" }
    """
    if text_col not in df.columns:
        raise ValueError(f"df must contain column '{text_col}'.")
    texts = df[text_col].astype(str).tolist()
    texts = [t for t in texts if t.strip()]
    if len(texts) < 50:
        raise ValueError("Not enough non-empty texts for CLM pretraining (recommend >= 50).")
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    # Split
    ds = Dataset.from_dict({text_col: texts})
    ds = ds.train_test_split(test_size=(1.0 - train_val_split), seed=seed)
    train_ds, eval_ds = ds['train'], ds['test']
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    # GPT2 has no default pad token, we use the eos_token for the text representation here:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Causal LM model for next-token prediction (not MLM!)
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    # LoRA targets for GPT2:
    if target_modules is None:
        target_modules = ["c_attn", "c_proj"]  # GPT2.
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj"]  # Right here
    )
    model = get_peft_model(model, lora_config)
    _print_trainable_params(model)
    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    # Causal LM collator (shifts labels for next-token prediction)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )#shifts labels for the next-token prediction here.
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        fp16=bool(use_cuda),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if eval_steps is not None else None,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    # Save LoRA adapters + tokenizer for the gpt model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {"model": model, "tokenizer": tokenizer, "trainer": trainer}




def ft_gpt2_clm(
    df: pd.DataFrame,
    text_col: str = "text",
    base_model_id: str = "gpt2",
    output_dir: str = "./gpt2_clm_adapted",
    seed: int = 0,
    max_length: int = 128,
    # training hyperparams
    num_train_epochs: float = 1.0,
    learning_rate: float = 5e-5,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    train_val_split: float = 0.98,
) -> Dict[str, Any]:
    """
    Fine-Tuning GPT-2 via Causal Language Modeling (CLM) on a custom text corpus
    Args:
        df: DataFrame containing the text column
        text_col: Name of the column with text data
        base_model_id: HuggingFace model identifier (default: "gpt2")
        output_dir: Directory to save the adapted model
        seed: Random seed for reproducibility
        max_length: Maximum sequence length
        num_train_epochs: Number of training epochs
        learning_rate: Learning rate (default 5e-5, standard for GPT-2 fine-tuning)
        per_device_train_batch_size: Batch size per device
        gradient_accumulation_steps: Gradient accumulation steps
        weight_decay: Weight decay for optimizer
        warmup_ratio: Warmup ratio for learning rate scheduler
        logging_steps: Logging frequency (steps)
        save_steps: Save checkpoint frequency (steps)
        eval_steps: Evaluation frequency (steps); if None, no evaluation
        train_val_split: Fraction of data for training (rest for validation)
    Returns:
        Dict with keys: model, tokenizer, trainer to save into another repo.
    """
    if text_col not in df.columns:
        raise ValueError(f"DataFrame must contain column '{text_col}'.")
    texts = df[text_col].astype(str).tolist()
    texts = [t for t in texts if t.strip()]  # filter empty strings
    if len(texts) < 50:
        raise ValueError(f"Not enough non-empty texts for CLM pretraining (got {len(texts)}, recommend >= 50).")
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    ds = Dataset.from_dict({text_col: texts})
    ds = ds.train_test_split(test_size=1 - train_val_split, seed=seed)
    train_ds, eval_ds = ds["train"], ds["test"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no default pad token
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",  # rectangular batches; consistent masking
        )
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        fp16=bool(use_cuda),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if eval_steps is not None else None,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {"model": model, "tokenizer": tokenizer, "trainer": trainer}

def ft_bert_mlm(
    df: pd.DataFrame,
    text_col: str = "text",
    base_model_id: str = "bert-base-uncased",
    output_dir: str = "./bert_mlm_adapted",
    seed: int = 0,
    max_length: int = 128,
    mlm_probability: float = 0.15,
    # training hyperparams
    num_train_epochs: float = 1.0,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    train_val_split: float = 0.98,
) -> Dict[str, Any]:
    if text_col not in df.columns:
        raise ValueError(f"df must contatn column '{text_col}'.")
    texts = df[text_col].astype(str).tolist()
    tests = [t for t in texts if t.strip()]
    if len(texts) < 50:
        raise ValueError("Not enough non-empty texts for MLM pretraining (recommend >= 50).")
    set_seed(seed)
    os.makedirs(output_dir, exist_ok = True)
    ds = Dataset.from_dict({text_col: texts})
    ds = ds.train_test_split(test_size = 1 - train_val_split)
    train_ds, eval_ds = ds["train"], ds["test"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    model = AutoModelForMaskedLM.from_pretrained(base_model_id)
    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",  # keeps batches rectangular; consistent masking
        )
    train_tok = train_ds.map(tokenize_fn, batched = True, remove_columns = [text_col])
    eval_tok = eval_ds.map(tokenize_fn, batched = True, remove_columns = [text_col])
    #Data Collators here:
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        fp16=bool(use_cuda),      # mixed precision if on GPU
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if eval_steps is not None else None,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {"model": model, "tokenizer": tokenizer, "trainer": trainer}


def ft_bert_mlm_lora(
    df: pd.DataFrame,
    text_col: str = "text",
    base_model_id: str = "bert-base-uncased",
    output_dir: str = "./bert_mlm_lora_adapted",
    seed: int = 0,
    # Tokenization
    max_length: int = 128,
    mlm_probability: float = 0.15,
    # LoRA hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    # Training hyperparams
    num_train_epochs: float = 1.0,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    train_val_split: float = 0.98,
) -> Dict[str, Any]:
    """
    Fine-Tuning (MLM) with LoRA adapters on df[text_col].
    Saves:
      - LoRA adapters to output_dir (model.save_pretrained)
      - tokenizer to output_dir
    Returns:
      dict with { "model", "tokenizer", "trainer" }
    """
    if text_col not in df.columns:
        raise ValueError(f"df must contain column '{text_col}'.")
    texts = df[text_col].astype(str).tolist()
    texts = [t for t in texts if t.strip()]
    if len(texts) < 50:
        raise ValueError("Not enough non-empty texts for MLM pretraining (recommend >= 50).")
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    # Split
    ds = Dataset.from_dict({text_col: texts})
    ds = ds.train_test_split(test_size=(1.0 - train_val_split), seed=seed)
    train_ds, eval_ds = ds["train"], ds["test"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    # Masked LM head model for MLM training
    model = AutoModelForMaskedLM.from_pretrained(base_model_id)
    if target_modules is None:
        target_modules = ["query", "value"]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        fp16=bool(use_cuda),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if eval_steps is not None else None,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    # Save LoRA adapters + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {"model": model, "tokenizer": tokenizer, "trainer": trainer}


def ft_bert_mlm_lora(
    df: pd.DataFrame,
    text_col: str = "text",
    base_model_id: str = "bert-base-uncased",
    output_dir: str = "./bert_mlm_lora_adapted",
    seed: int = 0,
    # Tokenization
    max_length: int = 128,
    mlm_probability: float = 0.15,
    # LoRA hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
    # Training hyperparams
    num_train_epochs: float = 1.0,
    learning_rate: float = 1e-4,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    logging_steps: int = 50,
    save_steps: int = 500,
    eval_steps: Optional[int] = None,
    train_val_split: float = 0.98,
) -> Dict[str, Any]:
    """
    Continued pretraining (MLM) with LoRA adapters on df[text_col].
    Saves:
      - LoRA adapters to output_dir (model.save_pretrained)
      - tokenizer to output_dir
    Returns:
      dict with { "model", "tokenizer", "trainer" }
    """
    if text_col not in df.columns:
        raise ValueError(f"df must contain column '{text_col}'.")
    texts = df[text_col].astype(str).tolist()
    texts = [t for t in texts if t.strip()]
    if len(texts) < 50:
        raise ValueError("Not enough non-empty texts for MLM pretraining (recommend >= 50).")
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    # Split
    ds = Dataset.from_dict({text_col: texts})
    ds = ds.train_test_split(test_size=(1.0 - train_val_split), seed=seed)
    train_ds, eval_ds = ds["train"], ds["test"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    # Masked LM head model for MLM training
    model = AutoModelForMaskedLM.from_pretrained(base_model_id)
    if target_modules is None:
        target_modules = ["query", "value"]
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    eval_tok = eval_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        save_total_limit=2,
        fp16=bool(use_cuda),
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok if eval_steps is not None else None,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    trainer.train()
    # Save LoRA adapters + tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {"model": model, "tokenizer": tokenizer, "trainer": trainer}
####


@torch.no_grad()
def gpt2_mean_pool_embeddings(
    texts, model_name = 'gpt2',
    batch_size = 32, max_length = 128):
    texts = texts.astype(str).tolist() if isinstance(texts, pd.Series) else [str(t) for t in texts]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is None:
        model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: (i + batch_size)]
        inputs = tokenizer(
            batch_texts, padding = True,
            truncation = True, max_length = max_length,
            return_tensors = 'pt').to(device)
        with torch.no_grad():
            out = model(**inputs)
        last_hidden = out.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()#(batch_size * S * H)
        #Mean pooling Logic:
        summed = torch.sum(last_hidden * mask, dim = 1)#(batch_size * S * H)
        counts = torch.clamp(mask.sum(dim = 1), min = 1e-9)
        mean_pooled = (summed/counts).cpu()#(Batch_size * H)
        all_embs.append(mean_pooled)#
    return torch.cat(all_embs, dim = 0).numpy()


#GPT/Bert Embedding with LORA, notice that because of the change of the versions, the AutoModelForCausalLM are switched from the previous version.
def gpt_embedding_with_lora(
    texts, lora_dir, base_model_id = 'gpt2',
    batch_size = 32, max_length = 128):
    '''
    Getting the contextual word embedding from LoRA
    '''
    if isinstance(texts, pd.Series):
        texts = texts.astype(str).tolist()
    else:
        texts = [str(t) for t in texts]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast = True)
    tokenizer.pad_token = tokenizer.eos_token
    base_encoder = AutoModelForCausalLM.from_pretrained(
        base_model_id, output_hidden_states = True
    )
    model = PeftModel.from_pretrained(base_encoder, lora_dir)
    model = model.to(device)
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        hidden_states = out.hidden_states
        last_hidden_state = hidden_states[-1]
        attn = inputs['attention_mask']
        mask = attn.unsqueeze(-1).expand(last_hidden_state.size()).float()
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts        # (B, H)
        all_embs.append(mean_pooled.cpu())
    return torch.cat(all_embs, dim = 0).numpy()

def bert_mean_pool_embeddings_with_lora(
    texts: Union[List[str], pd.Series],
    base_model_id: str,
    lora_dir: str,
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Mean-pooled BERT embeddings with LoRA adapters applied.

    Returns: (n_texts, hidden_size) numpy array (hidden_size=768 for bert-base-uncased).
    """
    if isinstance(texts, pd.Series):
        texts = texts.astype(str).tolist()
    else:
        texts = [str(t) for t in texts]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(lora_dir, use_fast=True)
    base_model = AutoModel.from_pretrained(base_model_id)
    model = PeftModel.from_pretrained(base_model, lora_dir)
    model = model.to(device)
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        last_hidden = out.last_hidden_state  # (B, L, H)
        attn = inputs["attention_mask"]      # (B, L) return to the last hidden state.
        mask = attn.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts        # (B, H)
        all_embs.append(mean_pooled.cpu())
    return torch.cat(all_embs, dim=0).numpy()

@torch.no_grad()
def bert_mean_pool_embeddings(
    texts: Union[List[str], pd.Series],
    model_name: str = "bert-base-uncased",
    batch_size: int = 32,
    max_length: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Mean-pooled embeddings from BERT last_hidden_state using attention mask.

    Returns: (n_texts, hidden_size) numpy array.
    """
    if isinstance(texts, pd.Series):
        texts = texts.astype(str).tolist()
    else:
        texts = [str(t) for t in texts]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out = model(**inputs)
        last_hidden = out.last_hidden_state
        attn = inputs["attention_mask"]
        # mean pooling with mask
        mask = attn.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = (last_hidden * mask).sum(dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        mean_pooled = summed / counts
        all_embs.append(mean_pooled.detach().cpu())
    return torch.cat(all_embs, dim=0).numpy()


@torch.no_grad()
def gpt2_mean_pool_embeddings(
    texts, model_name = 'gpt2',
    batch_size = 32, max_length = 128):
    texts = texts.astype(str).tolist() if isinstance(texts, pd.Series) else [str(t) for t in texts]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i: (i + batch_size)]
        inputs = tokenizer(
            batch_texts, padding = True,
            truncation = True, max_length = max_length,
            return_tensors = 'pt').to(device)
        with torch.no_grad():
            out = model(**inputs)
        last_hidden = out.last_hidden_state
        mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden.size()).float()
        #Mean pooling Logic:
        summed = torch.sum(last_hidden * mask, dim = 1)
        counts = torch.clamp(mask.sum(dim = 1), min = 1e-9)
        mean_pooled = (summed/counts).cpu()
        all_embs.append(mean_pooled)
    return torch.cat(all_embs, dim = 0).numpy()





def fine_tuning_data_wrapper_gpt(df, text_col = 'text',
    model_name = 'gpt2',
    output_dir_lora = './gpt_mlm_lora_adapted',
    output_dir = './gpt_mlm_lora_adapted',
    num_train_epochs_mlm = 6, num_train_epochs_lora = 3,
    max_length = 128, lora_r = 16, lora_alpha = 32, lora_dropout = 0.1,
    target_modules = ['query', 'value'],
    batch_size = 32, output_csv_emb = 'X_emb_semEval2022_bertFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_bertFineTuningLora.csv', column_to_join = None):
    res = ft_gpt2_mlm_lora(
        df, text_col = text_col, base_model_id = model_name,
        output_dir = output_dir_lora, num_train_epochs = num_train_epochs_mlm,
        max_length = max_length, lora_r = lora_r,
        lora_alpha = lora_alpha, lora_dropout = lora_dropout,
        target_modules = ['query', 'value']
    )
    X_emb_lora = gpt_embedding_with_lora(
        df[text_col],
        lora_dir = output_dir_lora,
        base_model_id = model_name,
        batch_size = batch_size, max_length = max_length
    )
    result = ft_gpt2_clm(df, text_col = text_col,
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
            X_emb_lora[col] = df[col]
        X_emb_lora.to_csv(output_csv_lora)



def fine_tuning_data_wrapper_bert(df, text_col = 'text',
    model_name = 'bert-base-uncased',
    lora_dir = './bert_mlm_lora_adapted',
    output_dir = './bert_mlm_adapted',
    output_dir_lora = './bert_mlm_lora_adapted',
    num_train_epochs_mlm = 1, num_train_epochs_lora = 1,
    max_length = 128, lora_r = 16, lora_alpha = 32, lora_dropout = 0.1,
    target_modules = ['query', 'value'],
    batch_size = 32, output_csv_emb = 'X_emb_semEval2022_bertFineTuning.csv',
    output_csv_lora = 'X_emb_semEval2022_bertFineTuningLora.csv', column_to_join = None):
    res = ft_bert_mlm_lora(
        df, text_col = text_col, base_model_id = model_name,
        output_dir = output_dir_lora, num_train_epochs = num_train_epochs_mlm,
        max_length = max_length, lora_r = lora_r,
        lora_alpha = lora_alpha, lora_dropout = lora_dropout,
        target_modules = ['query', 'value']
    )
    X_emb_lora = bert_mean_pool_embeddings_with_lora(
        df[text_col], base_model_id = model_name, lora_dir = output_dir_lora,
        batch_size = batch_size, max_length = max_length
    )
    result = ft_bert_mlm_lora(df, text_col = text_col,
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
            X_emb_lora[col] = df[col]
        X_emb_lora.to_csv(output_csv_lora)


def _print_trainable_params(model) -> None:
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100.0 * trainable / total if total else 0.0
    print(f"Trainable params: {trainable:,} / {total:,} ({pct:.4f}%)")










