import pandas as pd
import numpy as np
import torch
import os

df = pd.read_json('datasets/adnioasis/datalist_unet_InstructPix2Pix.json')

# convert to csv and rename columns to match the other datasets
df = df.drop(['Subject', 'ID'], axis=1)
df = df.rename(columns={'slice': 'path'})
df = df.rename(columns={'Age': 'age'})
df = df.rename(columns={'Sex': 'sex'})
df = df.rename(columns={'Diagnosis': 'diagnosis'})

# get min and max age
min_age = df['age'].min()
max_age = df['age'].max()
print(f"Min age: {min_age}, Max age: {max_age}")

# convert M to 0 and F to 1 in Sex column
df['sex'] = df['sex'].replace({'M': 0, 'F': 1})

# separate validation set
df_0 = df[df['diagnosis'] == 0].sample(n=50)
df_05 = df[df['diagnosis'] == 0.5].sample(n=50)
df_1 = df[df['diagnosis'] == 1].sample(n=50)
df = df.drop(df_0.index)
df = df.drop(df_05.index)
df = df.drop(df_1.index)
df_val = pd.concat([df_0, df_05, df_1]).reset_index(drop=True)

# separate test set
df_0 = df[df['diagnosis'] == 0].sample(n=50)
df_05 = df[df['diagnosis'] == 0.5].sample(n=50)
df_1 = df[df['diagnosis'] == 1].sample(n=50)
df = df.drop(df_0.index)
df = df.drop(df_05.index)
df = df.drop(df_1.index)
df_test = pd.concat([df_0, df_05, df_1]).reset_index(drop=True)

# train set
df_train = df.reset_index(drop=True)

# save the dataframes as a csv files
csv_folder = 'datasets/adnioasis/ADNI_OASIS_csv'
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
df_val.to_csv(os.path.join(csv_folder, 'valid.csv'))
df_test.to_csv(os.path.join(csv_folder, 'test.csv'))
df_train.to_csv(os.path.join(csv_folder, 'train.csv'))