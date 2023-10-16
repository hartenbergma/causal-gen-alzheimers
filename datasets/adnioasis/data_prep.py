import pandas as pd
import numpy as np

df = pd.read_json('datasets/adnioasis/datalist_unet_InstructPix2Pix.json')

# convert to csv and rename columns to match the other datasets
df.to_csv('datalist_unet_instructPix2Pix.csv')
df = df.drop(['Subject', 'ID'], axis=1)
df = df.rename(columns={'Age': 'age'})
df = df.rename(columns={'Sex': 'sex'})
df = df.rename(columns={'Diagnosis': 'diagnosis'})


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
df_val.to_csv('datasets/adnioasis/ADNI_OASIS_csv/valid.csv')
df_test.to_csv('datasets/adnioasis/ADNI_OASIS_csv/test.csv')
df_train.to_csv('datasets/adnioasis/ADNI_OASIS_csv/train.csv')