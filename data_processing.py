# Embedding Preprocessing
import numpy as np
import pandas as pd
import re
import random
from tqdm import tqdm

def pairs_to_dataset(path, length, number):
    df = pd.read_csv(path, low_memory=False).astype('string') # Load data
    if number < len(df['A']):
        df = df[:number]
    for i in range(len(df['A'])):
        df['Label'].iloc[i] = '1'
    df['Label'] = df['Label'].astype('int')
    # Process by sequence size
    drops = []
    for i in tqdm(range(len(df['A'])), desc='Trim Length'):
        #df['Label'].iloc[i] = str(1)
        try:
            if (len(df['SeqA'].iloc[i]) > length) or (len(df['SeqB'].iloc[i]) > length):
                drops.append(i)
        except:
            drops.append(i)

    df.drop(df.index[drops])

    len_df = len(df['A'])  # Keep original length
    df = pd.concat([df] * 2, ignore_index=True)  # Double size

    for i in range(len_df, 2 * len_df):  # Replace half of labels with 0
        df['Label'].iloc[i] = 0

    drops_shuf = []
    for i in tqdm(range(len(df['A'])), desc='Shuffle'):  # Shuffle sequences after first 3
        idA = df['A'].iloc[i]
        idB = df['B'].iloc[i]
        try:
            if (df['Label'].iloc[i] == 0):  # only shuffle non-interactor label
                if (i % 2 == 0):
                    SeqA = df['SeqA'].iloc[i]
                    df['SeqA'].iloc[i] = SeqA[:3] + ''.join(random.sample(SeqA[3:], len(SeqA[3:])))
                    df['A'].iloc[i] = 'A_' + idA + '_shuf-' + str(i)
                    df['B'].iloc[i] = 'B_' + idB + '-' + str(i)  # add unique matching identifier to pairs
                else:
                    SeqB = df['SeqB'].iloc[i]
                    df['SeqB'].iloc[i] = SeqB[:3] + ''.join(random.sample(SeqB[3:], len(SeqB[3:])))
                    df['B'].iloc[i] = 'B_' + idB + '_shuf-' + str(i)
                    df['A'].iloc[i] = 'A_' + idA + '-' + str(i)
            else:
                df['A'].iloc[i] = 'A_' + idA + '-' + str(i)
                df['B'].iloc[i] = 'B_' + idB + '-' + str(i)
        except:
            drops_shuf.append[i]

    df.drop(df.index[drops_shuf])

    drops_comb = []

    for i in tqdm(range(len(df['A'])), desc='Combine'):  # remove special aminos and add spaces
        SeqA = str(df['SeqA'].iloc[i])
        SeqA = re.sub(r'[UZOB]', 'X', SeqA)
        SeqA = ' '.join(list(SeqA))
        SeqB = str(df['SeqB'].iloc[i])
        SeqB = re.sub(r'[UZOB]', 'X', SeqB)
        SeqB = ' '.join(list(SeqB))
        df['SeqA'].iloc[i] = SeqA
        df['SeqB'].iloc[i] = SeqB
        df['Combined'].iloc[i] = '[CLS] ' + SeqA + ' [SEP] ' + SeqB + ' [SEP]'
        if len(df['Combined'].iloc[i]) < 30:
            drops_comb.append(i)

    df.drop(df.index[drops_comb])

    df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe in place
    df.to_csv(str(length) + 'labels_combined' + str(number) + '.csv')

pairs_to_dataset('PPI_seqs_trimmed.csv', 500, 20000000)