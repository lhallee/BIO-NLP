import pandas as pd
import torch
from datetime import datetime
from transformers import BertTokenizer

df = pd.read_csv('384MLMtraining.csv').astype('string')
Seqs = list(df['Combined'])
prot_tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')

inputs = prot_tokenizer(Seqs, return_tensors='pt', max_length=1027, truncation=True, padding='max_length')
torch.save(inputs, '384MLMtraining.pt')
