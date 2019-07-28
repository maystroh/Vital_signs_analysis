
import tqdm
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

list_columns = ['ID patient', 'Identifiant', 'Sexe', 'Age', 'Taille (cm)',
                'Poids réel (kg)', 'IMC (kg/m²)', 'Poids idéal théorique (kg)', 'IGS2',
                'Antécédents cardiaques', 'Antécédents cardiaques (détail)',
                'Antécédents respiratoires', 'Antécédents respiratoires (détail)',
                'Antécédents chirurgicaux', 'Antécédents chirurgicaux (détail)',
                'Antécédents autres', 'Pathologie actuelle', 'Interface',
                'Type de ventilation', 'Débit O2 L/min', 'Mode ventilatoire',
                'FIO2 (%)', 'PEP cmH2O', 'Volume courant (ml/kg)',
                'FR réglée (cycles/min)', 'AI ou PI', 'Trigger inspiratoire (L/min)',
                'Trigger expiratoire (%)', 'ACTIF/HME', 'PaO2/FIO2', 'SpO2/FIO2',
                'FR moyenne (cycle/min) sur 2h', 'SpO2 moyenne (%)  sur 2h',
                'Asynchronies patient/ventilateur', 'Hémodynamique',
                'AC/FA ou arythmie soutenue', 'Amine', 'Antihypertenseurs',
                'Pouls (bpm)', 'TAS (mmHg)', 'TAD (mmHg)', 'TAM (mmHg)',
                'Remplissage (ml) sur les dernières 24h', 'Hypovolémie', 'Sédation',
                'Curarisation', 'Statut immunitaire',
                'Lactactes (mmol/L) sur les dernières 24h', 'Devenir ICU',
                'Durée Séjour ICU', 'DEVENIR J28', 'Durée Séjour Hôpital',
                'Devenir Hospitalisation', 'Température', 'Glasgow', 'Temps VI',
                'Temps VI+VNI', 'Temps VI+(VNI+)Optiflow', 'ECMO', 'Dialyses',
                'Commentaire', 'Pourquoi pas valable NEWS', 'Echelle APVU']


def pop_unused_cols(data, key):
    print(data.shape)
    # data.pop('patient_file')
    # data.pop('rangee_selected')
    # data.pop('Antécédents cardiaques')
    data.pop('Antécédents cardiaques (détail)')
    # data.pop('Antécédents respiratoires')
    data.pop('Antécédents respiratoires (détail)')
    # data.pop('Antécédents chirurgicaux')
    data.pop('Antécédents chirurgicaux (détail)')
    data.pop('Antécédents autres')
    data.pop('Débit O2 L/min')
    data.pop('Mode ventilatoire')
    data.pop('FIO2 (%)')
    data.pop('PEP cmH2O')
    data.pop('Volume courant (ml/kg)')
    data.pop('FR réglée (cycles/min)')
    data.pop('AI ou PI')
    data.pop('Trigger inspiratoire (L/min)')
    data.pop('Trigger expiratoire (%)')
    data.pop('ACTIF/HME')
    data.pop('PaO2/FIO2')
    data.pop('SpO2/FIO2')
    data.pop('TAS (mmHg)')
    data.pop('TAD (mmHg)')
    data.pop('Asynchronies patient/ventilateur')
    data.pop('Remplissage (ml) sur les dernières 24h')
    data.pop('Lactactes (mmol/L) sur les dernières 24h')
    data.pop('Taille (cm)')
    data.pop('Poids réel (kg)')
    data.pop('Poids idéal théorique (kg)')
    data.pop('IGS2') # facteur de gravite calcule apres le depart du patient
    data.pop('Durée Séjour ICU')
    data.pop('Durée Séjour Hôpital')
    data.pop('Identifiant')
    data.pop('Pathologie actuelle')
    data.pop('Commentaire')
    data.pop('Pourquoi pas valable NEWS')
    data.pop('Temps VI')
    data.pop('Temps VI+VNI')
    data.pop('Temps VI+(VNI+)Optiflow')
    data.pop('Echelle APVU')
    data.pop('IMC (kg/m²)')
    # print(data.shape)
    # Checking for missing data
    NAs = pd.concat([data.isnull().sum()], axis=1, keys=[key])
    NAs[NAs.sum(axis=1) > 0]
    # print(NAs)

    ###### This section to be double checked...
    # # Filling missing Age values with mean
    data['Age'] = data['Age'].fillna(data['Age'].mean())
    # data['IMC (kg/m²)'] = data['IMC (kg/m²)'].fillna(data['IMC (kg/m²)'].mean())
    data['FR moyenne (cycle/min) sur 2h'] = data['FR moyenne (cycle/min) sur 2h'].fillna(
        data['FR moyenne (cycle/min) sur 2h'].mean())
    data['SpO2 moyenne (%)  sur 2h'] = data['SpO2 moyenne (%)  sur 2h'].fillna(
        data['SpO2 moyenne (%)  sur 2h'].mean())
    data['Pouls (bpm)'] = data['Pouls (bpm)'].fillna(data['Pouls (bpm)'].mean())
    # data['TAS (mmHg)'] = data['TAS (mmHg)'].fillna(data['TAS (mmHg)'].mean())
    # data['TAD (mmHg)'] = data['TAD (mmHg)'].fillna(data['TAD (mmHg)'].mean())
    data['TAM (mmHg)'] = data['TAM (mmHg)'].fillna(data['TAM (mmHg)'].mean())

    data['AC/FA ou arythmie soutenue'] = data['AC/FA ou arythmie soutenue'].fillna('NON')
    data['Antihypertenseurs'] = data['Antihypertenseurs'].fillna('NON')

    data['Antécédents chirurgicaux'] = data['Antécédents chirurgicaux'].fillna('NON')
    data['Antécédents respiratoires'] = data['Antécédents respiratoires'].fillna('NON')

    data['Hémodynamique'] = data['Hémodynamique'].fillna('STABLE')
    data['Hémodynamique'] = data['Hémodynamique'].fillna('STABLE')
    data['Curarisation'] = data['Curarisation'].fillna('NON')
    data['Hypovolémie'] = data['Hypovolémie'].fillna('NON')

    #It's just a test to verify what happens if I add these variable with scant amount of data
    # data['Mode ventilatoire'] = data['Mode ventilatoire'].fillna('Nothing')
    # data['FIO2 (%)'] = data['FIO2 (%)'].fillna(0)
    # data['PEP cmH2O'] = data['PEP cmH2O'].fillna(0)
    # data['Volume courant (ml/kg)'] = data['Volume courant (ml/kg)'].fillna(0)
    # data['FR réglée (cycles/min)']= data['FR réglée (cycles/min)'].fillna(0)

    # Filling missing Embarked values with most common value
    # data['Durée Séjour ICU'] = data['Durée Séjour ICU'].fillna(data['Durée Séjour ICU'].mode()[0])
    # data['Durée Séjour Hôpital'] = data['Durée Séjour Hôpital'].fillna(data['Durée Séjour Hôpital'].mode()[0])

    data['Statut immunitaire'] = data['Statut immunitaire'].fillna(data['Statut immunitaire'].mode()[0])

    data['Température'] = data['Température'].fillna(data['Température'].mean())
    data['Glasgow'] = data['Glasgow'].fillna(data['Glasgow'].mean())

    data['ECMO'] = data['ECMO'].fillna('NON')
    data['Dialyses'] = data['Dialyses'].fillna('NON')

    NAs = pd.concat([data.isnull().sum()], axis=1, keys=['data'])
    NAs[NAs.sum(axis=1) > 0]
    # print(NAs)
    return data


def convert_caterogical_data(data):
    # Getting Dummies from all other categorical vars
    cat_columns = data.select_dtypes(['object']).columns
    # print(cat_columns)
    for col in cat_columns:
        for_dummy = data.pop(col)
        data = pd.concat([data, pd.get_dummies(for_dummy, prefix=col)], axis=1)
    return data

target_list = ['Devenir ICU', 'DEVENIR J28', 'Devenir Hospitalisation']

def process_metadata(data_pandas, label_encoding_categories=False):

    #************ Verification step ************
    patients = data_pandas['Identifiant']
    assert (len(patients.unique().tolist()) == 472), 'Dataset should at least contain 500 unique patients!!'

    train_cat = pop_unused_cols(data_pandas, 'Train')
    NAs = pd.concat([train_cat.isnull().sum()], axis=1, keys=['Train'])
    NAs[NAs.sum(axis=1) > 0]
    # print(NAs)
    print("************************* COLUMNS USED IN THIS TEST ********************* ")
    print(train_cat.columns.values)
    print("************************* ********************* ")
    numeric_cols = train_cat.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = train_cat.select_dtypes(include=['object']).columns.tolist()
    object_cols.remove('Dataset')
    object_cols.remove('patient_file')
    object_cols.remove('range_selected')
    # print(numeric_cols)
    # print(object_cols)

    if label_encoding_categories:
        for col in object_cols:
            train_cat[col] = train_cat[col].astype('category')
            train_cat[col] = train_cat[col].cat.codes
    else:
        for col in target_list:
            train_cat[col] = train_cat[col].astype('category')
            train_cat[col] = train_cat[col].cat.codes
        [object_cols.remove(x) for x in object_cols[:] if x in target_list]
        for i in object_cols:
            if train_cat[i].nunique() == 2:
                train_cat[i] = train_cat[i].astype('category')
                train_cat[i] = train_cat[i].cat.codes
            else:
                print(i, train_cat[i].nunique())

    x_train = train_cat[train_cat['Dataset'] == 'Train'].reset_index(drop=True)

    # zero mean normalization on the dataset
    for index, col in enumerate(numeric_cols):
        scalar = StandardScaler()
        scalar.fit(x_train[col].values.reshape(-1, 1))
        train_cat[col] = scalar.transform(train_cat[col].values.reshape(-1, 1))

    train_cat.to_csv(path_or_buf='./training_converted_label_encoding_data.csv', sep='!', index=False)

    return train_cat

class Dataset:

    signal_sampling_frequency = 125

    factor_to_multiply_with_interval_time = 15 #It's used when data_augmentation is True // 15*2.048 = 30.72 seconds per sample
    columns_to_remove = ['patient_file', 'range_selected', 'Devenir ICU', 'DEVENIR J28', 'Devenir Hospitalisation', 'Dataset']

    def __init__(self, file_path, column_to_classify, column_values, window_time=2.0, training=False, augment_data=False, window_analysis=False):
        self.dataset_path = file_path
        self.data_augmentation = augment_data
        self.column_to_classify = column_to_classify
        self.column_values = column_values
        self.is_training= training
        self.timewindow_analysis = window_analysis
        self.window_duration = window_time
        df = pd.read_csv(self.dataset_path, sep='|')
        self.preprocessed_df = process_metadata(df, label_encoding_categories=True)

    def load_training_dataset(self):
        print('*** Loading training dataset ***')
        return self.read_dataset_file('Train')

    def load_dev_dataset(self):
        print('*** Loading dev dataset ***')
        return self.read_dataset_file('Dev')

    def load_test_dataset(self):
        print('*** Loading testing dataset ***')
        return self.read_dataset_file('Test')

    def read_dataset_file(self, dataset_type):
        signal_x = []
        metadata_x = []
        signal_metadata_y = []

        data_df = self.preprocessed_df[self.preprocessed_df['Dataset'] == dataset_type].reset_index(drop=True)

        data_df = data_df.sample(frac=1).reset_index(drop=True)

        for index, row in tqdm.tqdm(data_df.iterrows(), total=data_df.shape[0]):
            # print(row['patient_file'] + ' ' + row['range_selected'] + ' ' + row[self.column_to_classify] + ' ' + row['Dataset'])

            file_data = np.fromfile(row['patient_file'].encode('utf8'), dtype='>u2')
            ranges = row['range_selected'].split(":")
            data_to_treat = file_data[int(ranges[0]):int(ranges[1])]

            signal_label = int(row[self.column_to_classify])

            for x in self.columns_to_remove:
                row.pop(x)

            window_nb_values = int(self.window_duration * self.signal_sampling_frequency)

            if self.data_augmentation:
                #Data augmentation means here to segment the zone we selected from the main signal into N equal segments (so we have for each patient, N entries in the dataset instead of only one)
                nb_values_segment = window_nb_values * self.factor_to_multiply_with_interval_time
                for ind in range(0,len(data_to_treat), nb_values_segment):

                    if ind == nb_values_segment * (len(data_to_treat) // nb_values_segment):
                        #last element: if duration is less than half of one segment duration, add it to the list
                        if (nb_values_segment / 2) > (len(data_to_treat) - ind):
                            continue
                        signal_x.append(data_to_treat[ind : len(data_to_treat)])
                    else:
                        signal_x.append(data_to_treat[ind : ind + nb_values_segment])

                    metadata_x.append(row.values)
                    if self.timewindow_analysis:
                        signal_metadata_y.append([signal_label] * self.factor_to_multiply_with_interval_time)
                    else:
                        signal_metadata_y.append([signal_label])

            else:
                if self.timewindow_analysis:
                    trunc_samp = window_nb_values * int(len(data_to_treat) / window_nb_values)
                    signal_x.append(data_to_treat[:trunc_samp])
                    # signal_metadata_y.append([signal_label] * num_labels)
                else:
                    signal_x.append(data_to_treat)
                #TODO: to be modified according to timewindow_analysis
                metadata_x.append(row.values)
                signal_metadata_y.append([signal_label])


            # if len(signal_x) > 39:
            #     break

        return signal_x, metadata_x, signal_metadata_y