#different layers for fine-tuning:



def pretrain_bert_mlm(
    df: pd.DataFrame, 
    text_col = 'text', base_model_id = 'bert-base-uncased',
    output_dis = './bert_mlm_adapted',
    seed = 2026, max_length = 256,
    mlm_probability = 0.2, num_train_epoch = 10,
    learning_rate = 2e-5, per_device_train_batch_size = 16,
    gradient_accumulation_steps = 1,
    weight_decay = weight_decay,
    warmup_ratio = warmup_ratio,
    logging_steps = logging_steps,
    save_steps = save_steps,
    eval_steps = eval_steps,
    train_val_split: float = 0.8
):
    if text_col not in df.columns:
        raise ValueError(f"df must contain columns '{text_col}'.")
    texts = df[text_col].astype(str).tolist()
    tests = [t for t in texts if t.strip()]
    if len(texts) < 50:
        raise ValueError('Not enough non-empty texts for MLM pretraining(recommend >= 50).')
    set_seed(seed)
    os.makedir(output_dir, exist_ok = True)
    ds = Dataset.from_dict({text_col: texts})
    ds = ds.train_test_split(test_size = 1 - train_val_split)
    train_ds, eval_ds = ds['train'], ds['test']
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fact = True)
    model = AutoModelForMaskedLM.from_pretrained(base_model_id)
    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation = True,
            max_length = max_length,
            padding = 'max_length'
        )
    train_tok = train_ds.map(tokenize_fn, batched = True, remove_columns = [text_col])
    eval_tok = eval_ds.map(tokenize_fn, batched = True, remove_columns = [text_col])
    data_collator = DataCollatorForLanguageModeling(
        tokenizer = tokenizer,
        mlm = True,
        max_length = max_length,
        padding = 'max_length'
        )
    use_cuda = torch.cuda.is_available()
    args = TrainingArguments(
        output_dir = output_dir,
        overwrite_output_dir = True,
        num_training_epochs = num_train_epochs,
        learning_rate = learning_rate,
        per_device_train_batch_size = per_device_train_batch_size,
        per_device_eval_batch_size = per_device_eval_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        weight_decay = weight_decay,
        warmup_ratio = warmup_ratio,
        logging_steps = logging_steps,
        save_total_limit = 4,
        fp16 = bool(use_cuda),
        report_to = 'none'
    )
    trainer = Trainer(
        model = model,
        args = args,
        train_dataset = train_tok,
        eval_datase = eval_tok if eval_steps is not None else None,
        data_collator = data_collator,
        tokenizer = tokenizer
    )
    tokenizer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    return {'model': model, 'tokenizer': tokenizer, 'trainer': trainer}


def load_bert_encoder_with_lora(base_model_id: str, lora_dir, device):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenzier.from_pretrained(lora_dir, use_fast = True)#From the pretrained model here.
    base_encoder = AutoModel.from_pretrained(base_model_id)
    encoder = PeftModel.from_pretrained(base_encoder, lora_dir)
    encoder = encoder.to(device)
    encoder.eval()
    return tokenizer, encoder

@torch.no_grad()
def bert_mean_pooling(texts, model_name, batch_size = 16, max_length = 256, device = 'cuda'):
    all_emb = []
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
    for i in range(0, len(texts), batch_size):
        batch_text = texts[i:(i + batch_size)]
        inputs_text = tokenizer(
            batch_text, return_tensors = 'pt',
            padding = True, truncation = True,
            max_length = max_length
        )
        inputs_text = {k: v.to(device) for k, v in inputs_text.items()}
        out = model(**inputs)
        last_hidden = out.last_hidden_state #(batch_size, max_length, d)
        attn = inputs_text['attention_mask'] #(batch_size, max_length)
        mask = attn.unsqueeze(-1).expand(last_hidden.size(2)).float()
        mask = attn.unsqueeze(-1).expand(last_hidden.size(2)).float()#expand the last dimension w.r.t. d times
        summed = (last_hidden * mask).sum(dim = 1)#(batch_size * d)
        counts = torch.clamp(mask.sum(dim = 1), min = 1e-9)
        mean_pooled = summed/counts
        all_emb.append(mean_pooed.detach().cpu())
    return torch.cat(all_embs, dim = 0).numpy()


df = pd.read_csv("media_bias_annotation.csv")
res = pretrain_bert_mlm_lora(
         df,
         text_col="text",
         base_model_id="bert-base-uncased",
         output_dir="./bert_mlm_lora_adapted",
         num_train_epochs=8,
         max_length=128,
         lora_r=16,
         lora_alpha=24,
         lora_dropout=0.05,
         target_modules=["query", "value"],  # common choice for BERT
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
pd.DataFrame(X_emb).to_csv("X_emb_media_bias_bertFineTuning.csv")
pd.DataFrame(X_emb_lora).to_csv("X_emb_media_bias_bertFineTuningLora.csv")

pd.DataFrame(X_emb_lora).to_csv("X_emb_media_bias_bertFineTuningLora.csv")
pd.DataFrame(X_emb).to_csv("X_emb_media_bias_bertFineTuning.csv")
lora_config = {
    'lora_r': 16,
    'lora_alpha': 24,
    'lora_dropout': 0.1,
    'max_length': 64
}
def fine_tuning_dataset(df, clm, lora_r = 16, lora_alpha = 24, lora_dropout = 0.05, target_modules = target_modules):
    lora_r = lora_config.get('lora_r')
    lora_alpha = lora_config.get('lora_alpha')
    lora_dropout = lora_config.get('lora_dropout')
    res = pretrain_bert_mlm_lora()





















