import pandas as pd

parts = ['train', 'test', 'dev']

a = 'Unnamed: 0'

for part in parts:
    print(part)
    df = pd.read_csv(f'./data-df-w/{part}.tsv', sep='\t')
    df1 = df.drop(a, axis=1)
    new = {
        '0': df['2'],
        '1': df['3'],
        '2': df['0'],
        '3': df['1'],
        '4': df['4'],
        '5': df['5'],
        '6': df['6'],
        '7': df['7']
    }
    pd.DataFrame(new).to_csv(f'./data/{part}.tsv', sep='\t', index=0, header=False)