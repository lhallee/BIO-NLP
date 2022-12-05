import torch
import pandas as pd
from datetime import datetime
from transformers import BertTokenizer, BertForMaskedLM
from tqdm import tqdm


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: val[idx].detach().clone() for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def bert_MLM_token_train(model_path,
                         tokenizer_path,
                         data_path,
                         save_path,
                         max_seq_len, epochs, batch_size, lr=1e-3, save=True):
    df = pd.read_csv(data_path).astype('string')
    Seqs = list(df['Combined'])
    prot_tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    model = BertForMaskedLM.from_pretrained(model_path)

    inputs = prot_tokenizer(Seqs, return_tensors='pt', max_length=2 * max_seq_len + 3,
                            truncation=True, padding='max_length')
    inputs['labels'] = inputs.input_ids.detach().clone()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    now = datetime.now()
    for epoch in range(epochs):
        rand = torch.rand(inputs.input_ids.shape)
        # create mask array
        mask_arr = (rand < 0.15) * (inputs.input_ids != 2) * \
                   (inputs.input_ids != 3) * (inputs.input_ids != 0)
        selection = []

        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 4

        dataset = BertDataset(inputs)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        total_loss = 0
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
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
        print(f'Average Epoch Loss {total_loss}')
        if save:
            model.save_pretrained(save_path + str(now) + 'local_prot_bert_MLM_model')
    return 'Done'


def bert_MLM_train(model_path, data_path, save_path, epochs, batch_size, lr=1e-3, save=True):
    model = BertForMaskedLM.from_pretrained(model_path)
    inputs = torch.load(data_path)
    inputs['labels'] = inputs.input_ids.detach().clone()
    # create random array of floats with equal dimensions to input_ids tensor
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()
    # initialize optimizer
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    now = datetime.now()
    for epoch in range(epochs):
        rand = torch.rand(inputs.input_ids.shape)
        # create mask array
        mask_arr = (rand < 0.15) * (inputs.input_ids != 2) * \
                   (inputs.input_ids != 3) * (inputs.input_ids != 0)
        selection = []

        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 4

        dataset = BertDataset(inputs)

        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        total_loss = 0
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
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
        print(f'Average Epoch Loss {total_loss}')
        if save:
            model.save_pretrained(save_path + str(now) + 'local_prot_bert_MLM_model')
    return 'Done'