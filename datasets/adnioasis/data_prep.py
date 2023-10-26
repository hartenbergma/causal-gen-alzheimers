import pandas as pd
import numpy as np
import torch
import os

df_train = pd.read_json('datasets/adnioasis/datalist_train_unet_InstructPix2Pix.json')
df_test = pd.read_json('datasets/adnioasis/datalist_test.json')

# # remove all rows with diagnosis 0.5
# df_train = df_train[df_train['Diagnosis'] != 0.5]

# convert to csv and rename columns to match the other datasets
for df in [df_train, df_test]:
    df.drop(['Subject', 'ID'], axis=1, inplace=True)
    df.rename(columns={'slice': 'path', 'Age': 'age', 'Sex': 'sex', 'Diagnosis': 'diagnosis'}, inplace=True)
    df['sex'].replace({'M': 0, 'F': 1}, inplace=True)

# get min and max age 
min_age = min(df_train['age'].min(), df_test['age'].min())
max_age = max(df_train['age'].max(), df_test['age'].max())
print(f"Min age: {min_age}, Max age: {max_age}")
# get log of age and find mean and std
log_age = np.log(pd.concat([df_train['age'], df_test['age']]))
mean_log_age = log_age.mean()
std_log_age = log_age.std()
print(f"Mean log age: {mean_log_age}, Std log age: {std_log_age}")
# Min age: 42.69, Max age: 97.09
# Mean log age: 4.295305147924922, Std log age: 0.13014071547419478

# separate validation set
df_val = pd.DataFrame()
for diagnosis in [0, 0.5, 1]:
    df_diag = df_train[df_train['diagnosis'] == diagnosis].sample(n=50)
    df_train = df_train.drop(df_diag.index)
    df_val = pd.concat([df_val, df_diag]).reset_index(drop=True)

# # separate test set
df_mci = df_train[df_train['diagnosis'] == 0.5].sample(n=100)
df_train = df_train.drop(df_mci.index)
df_test = pd.concat([df_test, df_mci]).reset_index(drop=True)

# train set
# make sure that all 3 diagnosis types are represented equally, remove the rest
min_count = min(df_train['diagnosis'].value_counts())
df_train = df_train.groupby('diagnosis').apply(lambda x: x.sample(min_count)).reset_index(drop=True) 
# df_train = df.reset_index(drop=True)

# save the dataframes as a csv files
csv_folder = 'datasets/adnioasis/balanced_no_mci'
if not os.path.exists(csv_folder):
    os.makedirs(csv_folder)
df_val.to_csv(os.path.join(csv_folder, 'valid.csv'))
df_test.to_csv(os.path.join(csv_folder, 'test.csv'))
df_train.to_csv(os.path.join(csv_folder, 'train.csv'))