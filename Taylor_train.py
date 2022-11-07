from distutils.command.clean import clean
from math import sqrt
import sklearn as sk
import statistics
import pandas as pd
import os 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.svm import OneClassSVM
import pickle
from numpy import float64, float32
from scipy.spatial import distance
import itertools






file_name = "CycleTimes.pkl"
with open(file_name, "rb") as open_file:
  cycle_times = pickle.load(open_file)

file_name = "historic_hamming_distance.pkl"
with open(file_name, "rb") as open_f:
  historic_hamming = pickle.load(open_f)


def read_CAN_trace(file):
    dataset = pd.read_csv(file, sep=',')
    return dataset

def load_dataset(folder):

    datasets = []
    for filename in os.listdir(folder):
        f = os.path.join(folder, filename)
        datasets.append(read_CAN_trace(f))

    dataset = pd.DataFrame(columns=datasets[0].columns)
    for d in datasets:
        dataset=pd.concat([dataset, d], ignore_index=True)
    return dataset

def make_dataset_id(paths):
  datasets = []
  for p in paths:
    datasets.append(read_CAN_trace(p))

  dataset = pd.DataFrame(columns=datasets[0].columns)
  for d in datasets:
    dataset=pd.concat([dataset, d], ignore_index=True)
  return dataset

def print_results(y_true, predicted):
  print(f'accuracy  --> {accuracy_score(y_true,predicted)}')
  print(f'Confusion Matrix --> \n {confusion_matrix(y_true,predicted)}')
  print(f'F-1 Score --> {f1_score(y_true, predicted)}')

def from_hex_to_float(val):
  val = float.fromhex(val)
  return val

def anomaly_for_OCSVM(val):
  if val == False:
    return 1
  return -1

def anomaly_for_OCSVM_inverse(val):
  if val == 1:
    return "Valido"
  return "Anomalia"

def dataset_preprocessing(dataset):
  dataset=dataset.dropna() 
  dataset['ANOMALY']=dataset['ANOMALY'].astype(bool)
  dataset['PAYLOAD_HEX']=dataset['PAYLOAD_HEX'].apply(from_hex_to_float)

  dataset['PAYLOAD_HEX']=dataset['PAYLOAD_HEX'].astype(float32)
  dataset=dataset.drop(columns=['DLC'])

  return dataset

def hamming_distance(bin1, bin2):
  if bin1 == bin2:
    return 0

  count = 0
  for i in range(64):
    if bin1[i] != bin2[i]:
      count += 1
  return (count)

def make_flows(dataset, cycle_time_historic, hamming_historic, stop_ts = 1, inc_ts = 0.5):
  max_timestamp = max(dataset.timestamp)


  start_ts = 0
  flows = []

  while stop_ts < max_timestamp:

    datasetTW = dataset[dataset.timestamp.between(start_ts, stop_ts)]
    IDs = datasetTW['CAN_ID'].unique()

  

    for id in  IDs:
      tmp_df = datasetTW[datasetTW.CAN_ID == id]
      if len(tmp_df) > 2:
        is_anomaly = tmp_df['ANOMALY'].unique()
        if len(is_anomaly) == 2:
          is_anomaly = True
        elif len(is_anomaly) == 1:
          is_anomaly = is_anomaly[0]
        else:
          print("Error in ANOMALY field")
          exit()

        tot_hamming_dist = 0
        variance_hamming_values = [] 
    

        for i in range(len(tmp_df.PAYLOAD_BIN) -1) :
          first_datafield = tmp_df.PAYLOAD_BIN.iloc[i]
          second_datafield = tmp_df.PAYLOAD_BIN.iloc[i +1]

          hamming_value = hamming_distance(first_datafield, second_datafield)
          tot_hamming_dist += hamming_value
          variance_hamming_values.append(hamming_value)

  
        mu_h = tot_hamming_dist / len(tmp_df)
        variance_h = statistics.variance(variance_hamming_values)

        tot_time_diff = 0
        variance_time_diff = []

        for i in range(len(tmp_df)-1):
          first_timestamp = tmp_df.timestamp.iloc[i]
          second_timestamp = tmp_df.timestamp.iloc[i + 1]

          diff_value = second_timestamp - first_timestamp
          tot_time_diff += diff_value
          variance_time_diff.append(diff_value)

        mu_t = tot_time_diff / len(tmp_df)
        variance_t = statistics.variance(variance_time_diff)

        try:
          T_h = (mu_h - hamming_historic[f'{id}']['mu']) / sqrt((variance_h/len(tmp_df)))
        except ZeroDivisionError:
          T_h = np.nan
        
        try:
          T_t = (mu_t - cycle_time_historic[f'{id}']['mu']) / sqrt((variance_t/len(tmp_df)))
        except ZeroDivisionError:
          T_t = np.nan

        flows.append({"ID": id, "Np" : len(tmp_df) ,"Muh": mu_h, "sigma^2_h" : variance_h, "Mut" : mu_t, "sigma^2_t": variance_t, "T_t": T_t, "T_h": T_h, "ANOMALY":is_anomaly})
    start_ts += inc_ts
    stop_ts += inc_ts

  return pd.DataFrame.from_dict(data=flows)

def load_pickle_file(file_name):
  with open(file_name, "rb") as open_file:
    f = pickle.load(open_file)
  return f

def save_pickle_file(file_name, var_to_dump ):
  with open(file_name, "wb") as open_file:
    pickle.dump(var_to_dump, open_file)


def test_best_five_features(dataframe_for_training: pd.DataFrame, dataframe_for_test : pd.DataFrame):
  ocsvm_classifier = OneClassSVM(gamma='auto')
  print("\nFeatures: mu_t\n")
  
  columns_to_drop = [
      ['ID', 'Np', 'Muh', 'sigma^2_h', 'sigma^2_t', 'T_t', 'T_h', 'ANOMALY'],
      ['ID', 'Muh', 'sigma^2_h', 'sigma^2_t', 'T_t', 'T_h', 'ANOMALY'],
      ['ID', 'Np' ,'Muh', 'sigma^2_h',  'T_t', 'T_h', 'ANOMALY'],
      ['ID',  'Muh', 'sigma^2_t', 'T_t', 'T_h', 'ANOMALY'],
      ['ID', 'Np', 'Muh',  'sigma^2_t', 'T_t', 'T_h', 'ANOMALY']

  ]

  used_features = ["mu_t", "mu_t, Np", "mu_t, variance_t", "mu_t, Np, variance_h", "mu_t, variance_h"]

  for i in range(5):
    
    X_train = dataframe_for_training.drop(columns = columns_to_drop[i])
    x_test = dataframe_for_test.drop(columns = columns_to_drop[i])

    print(f"\nInizio FIT con features: {used_features[i]}\n")
    ocsvm_classifier.fit(X_train)
    print("\nFine FIT\n")
    pred = ocsvm_classifier.predict(x_test)

    y_true = dataframe_for_test['ANOMALY']
    new_value = []

    for val in pred:
      new_value.append(anomaly_for_OCSVM_inverse(val))


    new_pred = np.array(new_value)
    new_y_true = dataframe_for_test['ANOMALY'].apply(anomaly_for_OCSVM_inverse)

    print(f"\nRisultati con features: {used_features[i]}\n")
    print(classification_report(new_y_true, new_pred, labels=["Valido", "Anomalia"]))



def train_with_five_first_best_features(dataframe_for_training: pd.DataFrame):
  ocsvm_classifier = OneClassSVM(gamma='auto')
  
  
  columns_to_drop = [
          
      ['ID', 'Np', 'Muh', 'sigma^2_h', 'sigma^2_t', 'ANOMALY'],
      ['ID', 'Muh', 'sigma^2_h', 'sigma^2_t', 'ANOMALY'],
      ['ID', 'Np' ,'Muh', 'sigma^2_h', 'ANOMALY'],
      ['ID',  'Muh', 'sigma^2_t', 'ANOMALY'],
      ['ID', 'Np', 'Muh',  'sigma^2_t','ANOMALY']
  ]

  used_features = ["mu_t", "mu_t__Np", "mu_t__variance_t", "mu_t__Np__variance_h", "mu_t__variance_h"]
  for i in range(5):    
    X_train = dataframe_for_training.drop(columns = columns_to_drop[i])

    print(f"\nInizio FIT con features: {used_features[i]}\n")
    ocsvm_classifier.fit(X_train)
    print("\nFine FIT\n")
    path_to_save = f"/home/emanuele/Documenti/TESI/codice/terzo_paper/classificatori/OTIDS_clean_02w_01i_OCSVM_{used_features[i]}.pkl"
    save_pickle_file(path_to_save, ocsvm_classifier)
    print(f"\nSAVE OTIDS_free_dataframe_02w_01i_OCSVM_{used_features[i]}---------------------------")

 


if __name__ == "__main__":
  print("OTIDS_free_dataframe_02w_01i.pkl\n")
  clean_dataframe = load_pickle_file("/OTIDS_free_dataframe_02w_01i.pkl")
  clean_dataframe['ANOMALY'] = clean_dataframe['ANOMALY'].apply(anomaly_for_OCSVM)
  print(clean_dataframe)
  train_with_five_first_best_features(clean_dataframe)
  