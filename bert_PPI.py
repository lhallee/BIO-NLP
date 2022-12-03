import pandas as pd
import torch
from datetime import datetime
from transformers import BertTokenizer, BertForNextSentencePrediction
from tqdm import tqdm


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].detach().clone() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

def bert_token_train(model_path, tokenizer_path, data_path, save_path, epochs, batch_size, lr=1e-3, save=True):
    df = pd.read_csv(data_path).astype('string')
    df['Label'] = df['Label'].astype('int')
    SeqsA=list(df['SeqA'])
    SeqsB=list(df['SeqB'])
    labels = list(df['Label'])

    prot_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForNextSentencePrediction.from_pretrained(model_path)

    inputs = prot_tokenizer(SeqsA, SeqsB, return_tensors='pt', max_length=2003, truncation=True, padding='max_length')
    inputs['labels'] = torch.LongTensor([labels]).T

    dataset = BertDataset(inputs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        total_loss = 0
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels
                            )
            # extract loss
            loss = outputs.loss
            total_loss += loss.item()
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        total_loss = round(total_loss / len(loop), 3)
        print(f'Average loss {total_loss}')


    if save:
        now = datetime.now()
        model.save_pretrained(save_path + str(now) + 'local_prot_bert_NSP_model')
        #prot_tokenizer.save_pretrained(str(now) + 'local_prot_bert_NSP_tokenizer')
        return str(now) + 'local_prot_bert_NSP_model'
    else:
        return 'Done'


def bert_train(model_path, data_path, save_path, epochs, batch_size, lr=1e-3, save=True):
    model = BertForNextSentencePrediction.from_pretrained(model_path)
    inputs = torch.load(data_path)


    dataset = BertDataset(inputs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        total_loss = 0
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels
                            )
            # extract loss
            loss = outputs.loss
            total_loss += loss.item()
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())
        total_loss = round(total_loss / len(loop), 3)
        print(f'Average loss {total_loss}')


    if save:
        now = datetime.now()
        model.save_pretrained(save_path + str(now) + 'local_prot_bert_NSP_model')
        #prot_tokenizer.save_pretrained(str(now) + 'local_prot_bert_NSP_tokenizer')
        return str(now) + 'local_prot_bert_NSP_model'
    else:
        return 'Done'


def make_tokens(tokenizer_path, data_path):
    df = pd.read_csv(data_path).astype('string')[:50000]
    df['Label'] = df['Label'].astype('int')
    SeqsA = list(df['SeqA'])
    SeqsB = list(df['SeqB'])
    labels = list(df['Label'])
    prot_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    print('Calculating Tokens')
    inputs = prot_tokenizer(SeqsA, SeqsB, return_tensors='pt', padding='max_length', max_length=2003)
    inputs['labels'] = torch.LongTensor([labels]).T
    torch.save(inputs, 'nlp_train_data_100000.pt')


def nsp_eval(model_path, tokenizer_path, data_path):
    df = pd.read_csv(data_path).astype('string')
    df['Label'] = df['Label'].astype('int')
    SeqsA = list(df['SeqA'])
    SeqsB = list(df['SeqB'])
    labels = list(df['Label'])

    prot_tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    model = BertForNextSentencePrediction.from_pretrained(model_path)

    inputs = prot_tokenizer(SeqsA, SeqsB, return_tensors='pt', max_length=2003, truncation=True, padding='max_length')
    inputs['labels'] = torch.LongTensor([labels]).T
    dataset = BertDataset(inputs)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    correct = 0
    loop = tqdm(loader, leave=True)
    model.to(device)
    with torch.no_grad():
        model.eval()
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids
                            )
            pred = torch.argmax(outputs.logits)
            if pred.float() == labels.float():
                correct += 1

    acc = 100 * correct / len(df['Label'])
    print(acc)
    return acc

#nsp_eval('local_prot_bert_bfd', 'Rostlab/prot_bert_bfd', '500labels_combined1000.csv')