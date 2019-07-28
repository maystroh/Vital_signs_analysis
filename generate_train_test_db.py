import pandas as pd
import numpy as np
import os
from collections import Counter

percentage_dev = 10
percentage_train = 60
percentage_test = 30

df = pd.read_csv('../data/Annotations/signal_annotation_GT.csv', sep='|')

print(df.index)

print('************************')

for column in df.columns:
    print('Patient\'s "' + column + '" not set: ' + str(df[column].isnull().sum()))
    # It's a trick to check for nan or none values in a column
    # print(df[df[column] != df[column]].to_string(columns=['Date', 'ID patient', 'Devenir Hospitalisation']))

# 'DEVENIR J28', 'Devenir ICU', 'AC/FA ou arythmie soutenue'
column_to_treat = 'Devenir ICU'
column_to_treat_values = ['DECEDE', 'VIVANT']

df_column = df[pd.notnull(df[column_to_treat])]

#print(df_column_query['patient_file'])

#####################################
count_row = df_column.shape[0]

died_hosp = df_column[df_column[column_to_treat] == 'DECEDE'].reset_index(drop=True)
alive_hosp = df_column[df_column[column_to_treat] == 'VIVANT'].reset_index(drop=True)

died_count = died_hosp.shape[0]
died_dev_number = int(percentage_dev * died_count / 100)
died_test_number = int(percentage_test * died_count / 100)
died_train_number = died_count - died_dev_number - died_test_number

alive_count = alive_hosp.shape[0]
alive_dev_number = int(percentage_dev * alive_count / 100)
alive_test_number = int(percentage_test * alive_count / 100)
alive_train_number = alive_count - alive_dev_number - alive_test_number

print('count_row ' + str(count_row))

print('Alive Train ' + str(alive_train_number))
print('Alive Test ' + str(alive_test_number))
print('Alive Dev ' + str(alive_dev_number))

print('Died Train ' + str(died_train_number))
print('Died Test ' + str(died_test_number))
print('Died Dev ' + str(died_dev_number))

print('Training ' + str(died_dev_number + alive_dev_number + alive_train_number + died_train_number))
print('Testing ' + str(alive_test_number + died_test_number))

died_dev_indices = np.random.choice(died_count, died_dev_number, replace=False)
died_dev_subset = died_hosp.loc[died_dev_indices]
died_dev_subset['Dataset'] = 'Dev'
died_withoutdev_subset = died_hosp.drop(df.index[died_dev_indices]).reset_index(drop=True)
# print(died_dev_subset)
died_test_indices = np.random.choice(died_count - died_dev_number, died_test_number, replace=False)
died_test_subset = died_withoutdev_subset.loc[died_test_indices]
died_test_subset['Dataset'] = 'Test'
# print(died_test_subset)
died_train_subset = died_withoutdev_subset.drop(df.index[died_test_indices]).reset_index(drop=True)
died_train_subset['Dataset'] = 'Train'
# print(died_train_subset)

alive_dev_indices = np.random.choice(alive_count, alive_dev_number, replace=False)
alive_dev_subset = alive_hosp.loc[alive_dev_indices]
alive_dev_subset['Dataset'] = 'Dev'
alive_withoutdev_subset = alive_hosp.drop(df.index[alive_dev_indices]).reset_index(drop=True)
# print(alive_dev_subset)
alive_test_indices = np.random.choice(alive_count - alive_dev_number, alive_test_number, replace=False)
alive_test_subset = alive_withoutdev_subset.loc[alive_test_indices]
alive_test_subset['Dataset'] = 'Test'
# print(alive_test_subset)
alive_train_subset = alive_withoutdev_subset.drop(df.index[alive_test_indices]).reset_index(drop=True)
alive_train_subset['Dataset'] = 'Train'
# print(alive_train_subset)

training_dataset = pd.concat([alive_train_subset, died_train_subset, alive_dev_subset, died_dev_subset])
testing_dataset = pd.concat([alive_test_subset, died_test_subset])

training_dataset.to_csv(path_or_buf='./gt/signal_annotation_training_{0}.csv'.format(column_to_treat.replace(' ', '_')), sep='|', index=False)
print(training_dataset)

testing_dataset.to_csv(path_or_buf='./gt/signal_annotation_testing_{0}.csv'.format(column_to_treat.replace(' ', '_')), sep='|', index=False)
print(testing_dataset)
#
# ########################################
#
duration_files = []
for index, row in df_column.iterrows():
    data_file = row['patient_file']
    data_id = row['Identifiant']
    selected_range = row['range_selected']
    interval_border = selected_range.split(':')
    nb_values = int(interval_border[1]) - int(interval_border[0])
    duration = nb_values * 8 * pow(10, -3)
    duration_files.append(duration)
    if (duration > 500):
        print(data_id + ' / ' + data_file + ' : ' +  str(duration))


print('Stats on \'Devenir Hospitalisation\' ' + str(Counter(df['Devenir Hospitalisation'])))
print('Stats on \'AC/FA ou arythmie soutenue\' ' + str(Counter(df['AC/FA ou arythmie soutenue'])))


print("Mean : " + str(np.mean(duration_files)) + " | std : " + str(np.std(duration_files)))