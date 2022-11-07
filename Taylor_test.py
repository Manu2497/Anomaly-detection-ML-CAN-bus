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

def anomaly_for_OCSVM(val):
  if val == False:
    return 1
  return -1

def anomaly_for_OCSVM_inverse(val):
  if val == 1:
    return "Valid"
  return "Anomaly"



def load_pickle_file(file_name):
  with open(file_name, "rb") as open_file:
    f = pickle.load(open_file)
  return f

def save_pickle_file(file_name, var_to_dump ):
  with open(file_name, "wb") as open_file:
    pickle.dump(var_to_dump, open_file)


def test_attack(dataframe_for_test : pd.DataFrame, classifiers):
  columns_to_drop = [
      ['ID', 'Np', 'Muh', 'sigma^2_h', 'sigma^2_t', 'ANOMALY'],
      ['ID', 'Muh', 'sigma^2_h', 'sigma^2_t', 'ANOMALY'],
      ['ID', 'Np' ,'Muh', 'sigma^2_h', 'ANOMALY'],
      ['ID',  'Muh', 'sigma^2_t', 'ANOMALY'],
      ['ID', 'Np', 'Muh',  'sigma^2_t', 'ANOMALY']

  ]

  used_features = ["mu_t", "mu_t, Np", "mu_t, variance_t", "mu_t, Np, variance_h", "mu_t, variance_h"]
  
  


  for i in range(5):
    x_test = dataframe_for_test.drop(columns = columns_to_drop[i])

    print(f"\nStarting TEST with features: {used_features[i]}\n")
   
    pred = classifiers[i].predict(x_test)

    y_true = dataframe_for_test['ANOMALY']
    new_value = []

    for val in pred:
      new_value.append(anomaly_for_OCSVM_inverse(val))


    new_pred = np.array(new_value)
    new_y_true = dataframe_for_test['ANOMALY'].apply(anomaly_for_OCSVM_inverse)
 
    print(f"\nResults with features: {used_features[i]}\n")
    print(classification_report(new_y_true, new_pred, labels=["Valid", "Anomaly"]))

if __name__ == "__main__":
    print("OTIDS_impersonation_dataframe_02w_01i.pkl\n")

    classifiers_paths = [
      "/OTIDS_clean_02w_01i_OCSVM_mu_t.pkl",
      "/OTIDS_clean_02w_01i_OCSVM_mu_t__Np.pkl",
      "/OTIDS_clean_02w_01i_OCSVM_mu_t__variance_t.pkl",
      "/OTIDS_clean_02w_01i_OCSVM_mu_t__Np__variance_h.pkl",
      "/OTIDS_clean_02w_01i_OCSVM_mu_t__variance_h.pkl",


    ]
  


    classifiers = []
    for p in classifiers_paths:
      classifiers.append(load_pickle_file(p))
    print(classifiers)

    dataframe_for_test = load_pickle_file("/OTIDS_impersonation_dataframe_02w_01i.pkl")
    print(dataframe_for_test[dataframe_for_test['ANOMALY'] == True])
    dataframe_for_test['ANOMALY'] = dataframe_for_test['ANOMALY'].apply(anomaly_for_OCSVM)

    test_attack(dataframe_for_test,classifiers)