import pandas as pd
import torch
from datetime import datetime
from transformers import BertTokenizer

'''
df = pd.read_csv('256MLMtraining.csv').astype('string')
Seqs = list(df['Combined'])
prot_tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')

inputs = prot_tokenizer(Seqs, return_tensors='pt', max_length=515, truncation=True, padding='max_length')
torch.save(inputs, '256MLMtraining.pt')
'''

