
import os
import pandas as pd
import numpy as np

def choose_one(x):
    return x.values[0]
def topk(x):
    return list(x)[:10000]

raw_df = pd.read_csv('omics/test.csv')
df = raw_df.groupby('compound').agg({'smiles':choose_one, 'npy_path':topk}).reset_index()

print(len(df),'***')

c = np.random.choice(df.compound.values, 50, replace=False)
select_df = raw_df.loc[raw_df.compound.isin(c)].reset_index(drop=True)
print(select_df.shape)

for i in range(len(select_df)):
    npy_path = select_df.loc[i, 'npy_path']
    os.system(f'cp {npy_path} /rhome/jiahua.rao/workspace/omics_infomax_submission/dataset/omics/imgs')

def new_path(x):
    s = [x.split('/')[-1]]
    s.insert(0, 'dataset/omics/imgs')
    return '/'.join(s)
select_df['npy_path'] = select_df['npy_path'].apply(new_path)
select_df.to_csv('dataset/omics/test.csv', index=False)
print('END')