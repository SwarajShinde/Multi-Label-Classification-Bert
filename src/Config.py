DEVICE = 'cuda'
bert_path = 'bert-base-uncased'
tokenizer = transformers.BertTokenizer.from_pretrained(bert_path,do_lower_case=True)
MAX_LEN = 64
train_batch_size = 8
valid_batch_size = 4

loss_fn = nn.CrossEntropyLoss().to(DEVICE)
epochs = 5
optimizer =  AdamW(model.parameters(), lr=2e-5, correct_bias=False)
total_steps = len(train_dataloader) * epochs

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)
