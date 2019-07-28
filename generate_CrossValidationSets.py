import pandas as pd
import numpy as np
import os
from collections import Counter
import math

df = pd.read_csv('../data/Annotations/signal_annotation_GT_latest.csv', sep='|')

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

count_row = df_column.shape[0]

nb_folds = 7
nb_folds_training = 5
nb_folds_testing = 2

count_per_fold = count_row / nb_folds

died_hosp = df_column[df_column[column_to_treat] == 'DECEDE'].reset_index(drop=True)
alive_hosp = df_column[df_column[column_to_treat] == 'VIVANT'].reset_index(drop=True)

died_count = died_hosp.shape[0]
percentage_died_per_fold = (died_count * 100) / count_row

percentage_alive_per_fold = 100 - percentage_died_per_fold
print('nb Patients ' + str(count_row))
print('nb alive {} '.format(alive_hosp.shape[0]))
print('nb died {} '.format(died_hosp.shape[0]))

print('For {} folds : count per fold {}'.format(nb_folds, count_per_fold))
print('% died computed per fold {}'.format(percentage_died_per_fold))
print('% alive computed per fold {}'.format(percentage_alive_per_fold))

died_number_per_fold = int(math.ceil((percentage_died_per_fold * count_per_fold) / 100))
alive_number_per_fold = int(math.floor((percentage_alive_per_fold * count_per_fold) / 100))

print('Died per fold ~ ' + str(died_number_per_fold))
print('Alive per fold ~ ' + str(alive_number_per_fold))

print('Training ~ ' + str(nb_folds_training * (died_number_per_fold+alive_number_per_fold)))
print('Testing ~ ' + str(nb_folds_testing * (died_number_per_fold+alive_number_per_fold)))

dev_percentage_per_fold = 15

died_hosp_iter = died_hosp.sample(frac=1).reset_index(drop=True)
alive_hosp_iter = alive_hosp.sample(frac=1).reset_index(drop=True)

indices_died_hosp = range(0,len(died_hosp_iter.index))
indices_alive_hosp = range(0,len(alive_hosp_iter.index))

for fold in range(1, nb_folds+1):
    print('\n **** decomposition ' + str(fold))
    died_lower_bound = (fold - 1) * died_number_per_fold
    died_upper_bound = (fold * died_number_per_fold) if fold < nb_folds else len(died_hosp_iter.index)
    print(died_lower_bound)
    print(died_upper_bound)
    died_test_indices = list(range(died_lower_bound, died_upper_bound))
    print(died_test_indices)
    died_test_subset = died_hosp_iter.iloc[died_test_indices]
    died_training_indices = [x for x in indices_died_hosp if x not in died_test_indices]
    print(died_training_indices)
    died_training_subset = died_hosp_iter.iloc[died_training_indices].sample(frac=1).reset_index(drop=True)
    died_nb_dev = len(died_training_subset.index) * dev_percentage_per_fold // 100
    died_dev_indices = np.random.choice(len(died_training_subset.index), died_nb_dev, replace=False)
    print(died_dev_indices)
    died_dev_subset = died_training_subset.iloc[died_dev_indices.tolist()]
    died_train_subset = died_training_subset.drop(died_dev_subset.index.values)
    print(died_train_subset.index.values)

    died_test_subset['Dataset'] = 'Test'
    died_dev_subset['Dataset'] = 'Dev'
    died_train_subset['Dataset'] = 'Train'

    print('died nb dev {}'.format(len(died_dev_subset.index)))
    print('died nb test {}'.format(len(died_test_subset.index)))
    print('died nb train {}'.format(len(died_train_subset.index)))

    alive_lower_bound = (fold - 1) * alive_number_per_fold
    alive_upper_bound = (fold * alive_number_per_fold) if fold < nb_folds else len(alive_hosp_iter.index)
    alive_test_indices = list(range(alive_lower_bound, alive_upper_bound))
    print(alive_test_indices)
    alive_test_subset = alive_hosp_iter.iloc[alive_test_indices]
    # print(alive_test_subset.index.values)
    alive_training_indices = [x for x in indices_alive_hosp if x not in alive_test_indices]
    print(alive_training_indices)
    alive_training_subset = alive_hosp_iter.iloc[alive_training_indices].sample(frac=1).reset_index(drop=True)
    alive_nb_dev = len(alive_training_subset.index) * dev_percentage_per_fold // 100
    alive_dev_indices = np.random.choice(len(alive_training_subset.index), alive_nb_dev, replace=False)
    print(alive_dev_indices)
    alive_dev_subset = alive_training_subset.iloc[alive_dev_indices]
    alive_train_subset = alive_training_subset.drop(alive_dev_subset.index.values)
    print(alive_train_subset.index.values)

    alive_dev_subset['Dataset'] = 'Dev'
    alive_test_subset['Dataset'] = 'Test'
    alive_train_subset['Dataset'] = 'Train'

    print('alive nb dev {}'.format(len(alive_dev_subset.index)))
    print('alive nb test {}'.format(len(alive_test_subset.index)))
    print('alive nb train {}'.format(len(alive_train_subset.index)))

    training_dataset = pd.concat([alive_train_subset, died_train_subset, alive_dev_subset, died_dev_subset, alive_test_subset, died_test_subset])
    training_dataset.to_csv(path_or_buf='./gt/signal_annotation_training_{0}_{1}.csv'.format(column_to_treat.replace(' ', '_'), fold), sep='|', index=False)

#
# nb_folds = 5
#
# #####################################
# count_row = df_column.shape[0]
#
# died_hosp = df_column[df_column[column_to_treat] == 'DECEDE'].reset_index(drop=True)
# alive_hosp = df_column[df_column[column_to_treat] == 'VIVANT'].reset_index(drop=True)
#
# died_count = died_hosp.shape[0]
# died_dev_number = int(percentage_dev * died_count / 100)
# died_train_number = died_count - died_dev_number
#
# alive_count = alive_hosp.shape[0]
# alive_dev_number = int(percentage_dev * alive_count / 100)
# alive_train_number = alive_count - alive_dev_number
#
# print('count_row ' + str(count_row))
#
# print('Alive Train ' + str(alive_train_number))
# print('Alive Dev ' + str(alive_dev_number))
#
# print('Died Train ' + str(died_train_number))
# print('Died Dev ' + str(died_dev_number))
#
# print('Training ' + str(died_dev_number + alive_dev_number + alive_train_number + died_train_number))
#
# died_hosp_iter = died_hosp.sample(frac=1).reset_index(drop=True)
# alive_hosp_iter = alive_hosp.sample(frac=1).reset_index(drop=True)
#
# for fold in range(1, nb_folds+1):
#
#     print(fold)
#     died_dev_subset = died_hosp_iter.loc[(fold-1) * died_dev_number: (fold * died_dev_number - 1) if fold < nb_folds else  len(died_hosp_iter.index)]
#     print(died_dev_subset.index.values)
#     died_train_subset = died_hosp.drop(died_dev_subset.index.values).reset_index(drop=True)
#     died_dev_subset['Dataset'] = 'Dev'
#     died_train_subset['Dataset'] = 'Train'
#
#     alive_dev_subset = alive_hosp_iter.loc[(fold-1) * alive_dev_number: (fold * alive_dev_number - 1) if fold < nb_folds else  len(
#         alive_hosp_iter.index)]
#     print(alive_dev_subset.index.values)
#     alive_train_subset = alive_hosp.drop(alive_dev_subset.index.values).reset_index(drop=True)
#     alive_dev_subset['Dataset'] = 'Dev'
#     alive_train_subset['Dataset'] = 'Train'
#
#     training_dataset = pd.concat([alive_train_subset, died_train_subset, alive_dev_subset, died_dev_subset])
#
#     training_dataset.to_csv(
#         path_or_buf='./gt/signal_annotation_training_{0}_{1}.csv'.format(column_to_treat.replace(' ', '_'),fold), sep='|', index=False)
#     # print(training_dataset)


# ########################################
# #
# duration_files = []
# for index, row in df_column.iterrows():
#     data_file = row['patient_file']
#     data_id = row['Identifiant']
#     selected_range = row['range_selected']
#     interval_border = selected_range.split(':')
#     nb_values = int(interval_border[1]) - int(interval_border[0])
#     duration = nb_values * 8 * pow(10, -3)
#     duration_files.append(duration)
#     if (duration > 500):
#         print(data_id + ' / ' + data_file + ' : ' +  str(duration))
#
#
# print('Stats on \'Devenir Hospitalisation\' ' + str(Counter(df['Devenir Hospitalisation'])))
# print('Stats on \'AC/FA ou arythmie soutenue\' ' + str(Counter(df['AC/FA ou arythmie soutenue'])))
#
#
# print("Mean : " + str(np.mean(duration_files)) + " | std : " + str(np.std(duration_files)))