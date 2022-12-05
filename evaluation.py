import torch
import pandas as pd
from transformers import BertTokenizer, BertForNextSentencePrediction, BertForMaskedLM, pipeline
from tqdm import tqdm


def nsp_eval(model_path, tokenizer_path, data_path, max_seq_len, num):
    df = pd.read_csv(data_path).astype('string')[:num]
    df['Label'] = df['Label'].astype('int')
    SeqsA = list(df['SeqA'])
    SeqsB = list(df['SeqB'])
    labels = list(df['Label'])
    prot_tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    model = BertForNextSentencePrediction.from_pretrained(model_path)
    inputs = prot_tokenizer(SeqsA, SeqsB, return_tensors='pt', max_length=2 * max_seq_len +3 ,
                            truncation=True, padding='max_length')
    inputs['labels'] = torch.LongTensor([labels]).T
    dataset = BertDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
                            token_type_ids=token_type_ids)
            pred = torch.argmax(outputs.logits)
            if pred.float() == labels.float():
                correct += 1

    acc = 100 * correct / len(df['Label'])
    print(acc)
    return acc


def bert_MLM_eval(model_path, tokenizer_path, data_path, num):
    amino_list = 'LAGVESIKRDTPNQFYMHCWXUBZO'
    model = BertForMaskedLM.from_pretrained(model_path)
    prot_tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=False)
    df = pd.read_csv(data_path).astype('string')[:num]
    Seqs = list(df['Combined'])
    unmasker = pipeline('fill-mask', model=model, tokenizer=prot_tokenizer)
    correct = 0
    for i in range(num):
        seq = Seqs[i]
        masks = 0
        while masks < 1:
            m = random.randint(6, len(seq)-6)
            gt = seq[m]
            if seq[m] in amino_list and seq[m-1] == seq[m+1]:
                seq = seq[:m] + '[MASK]' + seq[m+1:]
                pred = unmasker(seq)
                if gt == list(pred[0].values())[2]:
                    correct += 1
                masks += 1
    acc = 100 * correct / num
    return acc