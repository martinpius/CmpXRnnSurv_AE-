# -*- coding: utf-8 -*-
"""MIMIC_III Preprocessing

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-Fq0mhadm0dLTWC2DUkSMMQvVh9_sPVJ
"""

from google.colab import drive, files, auth
drive.mount("/content/drive/", force_remount = True)
try:
  COLAB = True
  import pandas as pd
  import numpy as np
  import torch, time
  from collections import Counter
  from sklearn.preprocessing import LabelEncoder, MinMaxScaler
  import matplotlib.pyplot as plt
  import torch,os
  pd.set_option("display.max_columns", None)
  pd.set_option("display.max_rows", None)
  print(f">>>> You are on CoLaB with Pandas version {pd.__version__}")

  def __timefmt__(t: float = 123.178)->float:
    h = int(t / (60 * 60))
    m = int(t % (60 * 60) / 60)
    s = int(t % 60)
    return f"hrs: {h} min: {m:>02} sec: {s:>05.2f}"

except Exception as e:
  print(f">>>> {type(e)}: {e}\n>>>> Please corect {type(e)} and reload the drive")
  COLAB = False
print(f">>>> testing the time formating function...........\n>>>> time elapsed\t{__timefmt__()}")



data_path = "/content/drive/MyDrive/MIMIC3_BIG_QUERY"
fnames = ['icu_clean1', 'icu_clean2', 'icu_clean3', "mimic3_clear_martin"]

#auth.authenticate_user()
#%%bigquery --project strange-song-274510 MIMIC3_CLEAR
#SELECT *
#FROM `strange-song-274510.111.MIMIC3_CLEAR_MARTIN`



def df_to_tensor(df):
  return torch.from_numpy(df.values).to(torch.int64)



# Loading the data from the drive
def __readmyCSV__(data_path,fnames, chunksize = 10000):
  """
  @Author: Martin Pius
  --------------------
  -This method read the big CSV datafiles from the given path
  Parameters:
  -----------
  data_path: str: A path/URL contains the data files
  fnames: List: List of files 
  chunk_size: Int: A chunk size 

  Returns:
  ----------
  ICU_data: pd.DataFrame 
  """
  for k in range(len(fnames)):
    dfm = data_path +"/" + fnames[k] +".csv"
    icu_data = pd.read_csv(dfm, chunksize = chunksize)
    if fnames[k]== 'icu_clean1':
      ICU_data1 = pd.concat(icu_data)
    elif fnames[k] == "icu_clean2":
      ICU_data2 = pd.concat(icu_data)
    elif fnames[k] == "mimic3_clear_martin":
      ICU_data_clean = pd.concat(icu_data)
    else:
      ICU_data3 = pd.concat(icu_data)
  frames = [ICU_data1, ICU_data2, ICU_data3]
  return ICU_data_clean



def __NewValues__(data_values, data_keys):
  list1 = data_values
  list2 = data_keys
  new_values1 = []
  for k in range(len(list1)):
    if list1[k]==1:
      new_values1 = new_values1 + [1]
    elif list1[k] == 2:
      new_values1 = new_values1 + [2,2]
    elif list1[k] == 3:
      new_values1 = new_values1 + [3,3,3]
    elif list1[k] == 4:
      new_values1 = new_values1 + [4,4,4,4]
    elif list1[k] == 5:
      new_values1 = new_values1 + [5,5,5,5,5]
    elif list1[k] == 6:
      new_values1 = new_values1 + [6,6,6,6,6,6]
    elif list1[k] == 7:
      new_values1 = new_values1 + [7,7,7,7,7,7,7]
    elif list1[k] == 8:
      new_values1 = new_values1 + [8,8,8,8,8,8,8,8]
    elif list1[k] == 9:
      new_values1 = new_values1 + [9,9,9,9,9,9,9,9,9]
    elif list1[k] == 10:
      new_values1 = new_values1 + [10,10,10,10,10,10,10,10,10,10]
    elif list1[k] == 11:
      new_values1 = new_values1 + [11,11,11,11,11,11,11,11,11,11,11]
    elif list1[k] == 12:
      new_values1 = new_values1 + [12,12,12,12,12,12,12,12,12,12,12,12]
    elif list1[k] == 13:
      new_values1 = new_values1 + [13,13,13,13,13,13,13,13,13,13,13,13,13]
    elif list1[k] == 14:
      new_values1 = new_values1 + [14,14,14,14,14,14,14,14,14,14,14,14,14,14]
    elif list1[k] == 15:
      new_values1 = new_values1 + [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]
    elif list1[k] == 16:
      new_values1 = new_values1 + [16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16]
    elif list1[k] == 17:
      new_values1 = new_values1 + [17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17,17]
    elif list1[k] == 18:
      new_values1 = new_values1 + [18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18,18]
    elif list1[k] == 19:
      new_values1 = new_values1 + [19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19,19]
    elif list1[k] == 20:
      new_values1 = new_values1 + [20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20,20]
    elif list1[k] == 21:
      new_values1 = new_values1 + [21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21,21]
    elif list1[k] == 22:
      new_values1 = new_values1 + [22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22,22]
    elif list1[k] == 23:
      new_values1 = new_values1 + [23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23,23]
    elif list1[k] == 24:
      new_values1 = new_values1 + [24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24,24]
    elif list1[k] == 25:
      new_values1 = new_values1 + [25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25,25]
    elif list1[k] == 26:
      new_values1 = new_values1 + [26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26,26]
    elif list1[k] == 27:
      new_values1 = new_values1 + [27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27,27]
    elif list1[k] == 28:
      new_values1 = new_values1 + [28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28]
    elif list1[k] == 29:
      new_values1 = new_values1 + [29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29,29]
    elif list1[k] == 30:
      new_values1 = new_values1 + [30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30]
    elif list1[k] == 31:
      new_values1 = new_values1 + [31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31]
    elif list1[k] == 32:
      new_values1 = new_values1 + [32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]
    elif list1[k] == 33:
      new_values1 = new_values1 + [33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33]
    elif list1[k] == 34:
      new_values1 = new_values1 + [34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34,34]
    elif list1[k] == 35:
      new_values1 = new_values1 + [35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35,35]
    elif list1[k] == 36:
      new_values1 = new_values1 + [36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36,36]
    elif list1[k] == 37:
      new_values1 = new_values1 +[37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37]
    elif list1[k] == 38:
      new_values1 = new_values1 + [38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38,38]
    elif list1[k] == 39:
      new_values1 = new_values1 + [39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39,39]
    elif list1[k] == 40:
      new_values1 = new_values1 + [40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40,40]
    elif list1[k] == 41:
      new_values1 = new_values1 + [41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41,41]
    else:
      pass
  return new_values1



def MIMICIII_preprocess(data_path, files_names):
  """
  @Author: Martin Pius
  ---------------------
  -This Module pre-processing the mimic3.csv data file and return 
   the torch tensor dataset


  """
  print(f">>>> Please wait, this may take a while...........")
  ICU_data = __readmyCSV__(data_path = data_path, fnames = files_names)
  
  values = list(Counter(ICU_data.subject_id).values())
  keys = list(Counter(ICU_data.subject_id).keys())
  new_values1 = __NewValues__(values, keys)
  ICU_data['new_id_1'] = new_values1

  # grouping subjects according to recurrent times
  data1 = ICU_data.loc[ICU_data.new_id_1 == 1] # first recurrent
  data2 = ICU_data.loc[ICU_data.new_id_1 == 2] # second recurrent
  data3 = ICU_data.loc[ICU_data.new_id_1 == 3] # third recurrent
  data4 = ICU_data.loc[ICU_data.new_id_1 == 4] # forth recurrent
  data5 = ICU_data.loc[ICU_data.new_id_1 == 5] # fifth recurrent
  data1.set_index("subject_id", inplace = True)
  data2.set_index("subject_id", inplace = True)
  data3.set_index("subject_id", inplace = True)
  data4.set_index("subject_id", inplace = True)
  data5.set_index("subject_id", inplace = True)

  # padding the data with zero rows to match the shapes
  e1 = data4.reset_index().drop_duplicates(subset = 'subject_id')
  df_4 = pd.concat([data4.reset_index(), e1], axis = 0)
  df_4 = df_4.set_index("subject_id").sort_index()
  e2 = data3.reset_index().drop_duplicates(subset = "subject_id")
  e2 = pd.concat([e2[:]]*2, ignore_index=True)
  df_3 = pd.concat([data3.reset_index(), e2], axis = 0).set_index("subject_id").sort_index()
  e3 = data2.reset_index().drop_duplicates(subset = "subject_id")
  e3 = pd.concat([e3[:]]*3, ignore_index=True)
  df_2 = pd.concat([data2.reset_index(), e3], axis = 0).set_index("subject_id").sort_index()
  e1 = data1.reset_index().drop_duplicates(subset = "subject_id")
  e1 = pd.concat([e1[:]]*4, ignore_index=True)
  df_1 = pd.concat([data1.reset_index(), e1], axis = 0).set_index("subject_id").sort_index()
  # assembles all datafiles together
  mimic3_new = pd.concat([df_1, df_2, df_3, df_4, data5], axis = 0)
  
  #drop columns with excessive missing values
  for col in list(mimic3_new.columns):
    if mimic3_new[col].isna().sum()>10000:
      mimic3_new.drop(col, axis = 1, inplace = True) 
  
  #Impute the missing values [for categorical add one class "NA"]
  for col in list(mimic3_new.columns):
    if mimic3_new.dtypes[col] == "object":
      mimic3_new[col] = mimic3_new[col].fillna("NA")
    else:
      mimic3_new[col] = mimic3_new[col].fillna(mimic3_new[col].mean())
  
  # prepares the target variable for recurrent events with competing risks
  

  # defining the label variable
  target = mimic3_new.loc[:, ["los_icu","DIAGNOSIS","hospital_expire_flag"]]
  target["label"] = np.zeros(shape = (len(target),), dtype = np.int64) # place holder

  for k in range(len(target)):
    if target.reset_index()["DIAGNOSIS"][k] == 1 and target.reset_index()["hospital_expire_flag"][k] == 0:
      target["label"][k] = "heart problem"
    elif target.reset_index()["DIAGNOSIS"][k] == 0 and target.reset_index()["hospital_expire_flag"][k] == 0:
      target["label"][k] = "other risks"
    else:
      target["label"][k] = "censore"

  label = LabelEncoder().fit_transform(target["label"])
  label_dict =  {"heart problem": 1, "other risks": 2, "censore": 0}
  surv_time = target['los_icu']
  surv_tensor = torch.tensor(target['los_icu'].values)
  label_tensor = torch.tensor(label)
  surv_time_dict = Counter(surv_time)
  keys = np.array(list(surv_time_dict.keys()))
  values = np.array(list(surv_time_dict.values()))
  
  #for visualizing the distribution of survival time
  x, y = list(), list()
  for i in range(95):
    if keys[i] < 6 and keys[i] != 0:
      x.append(keys[i]), y.append(values[i])
  
  #Get the covariates matrix
  covariates = mimic3_new.drop(["los_icu","DIAGNOSIS","hospital_expire_flag", 
                              "intime", "outtime","icustay_id","new_id_1",
                              "hadm_id", "admittime","dischtime","depression", 
                              "alcohol_abuse", "blood_loss_anemia","cardiac_arrhythmias", 
                              "deficiency_anemias","drug_abuse","paralysis", 
                              "paralysis_1", "hypothyroidism","psychoses", "row_number", "aids"], axis = 1)
  
  dict_comp1 = {"gender": {"M": "Male","F": "Female",0.0: "pad"}}
  dict_comp2 = {"ethnicity_grouped":{0.0: "pad","white":"white",
              "unkown":"unknown","black":"black",
              "asian":"asian","hispanic":"hispanic","other":"other","native":"native"}}
  
  covariates.gender = covariates.gender.astype("category")
  covariates.ethnicity_grouped = covariates.ethnicity_grouped.astype("category")
  covariates = covariates.replace(dict_comp1)
  covariates = covariates.replace(dict_comp2)
  covariates.gender = covariates.gender.astype("object")
  covariates.ethnicity_grouped = covariates.ethnicity_grouped.astype("object")

  #Categorical embedding of categorical variables
  for col in list(covariates.columns):
    if covariates[col].dtypes == "object":
      covariates[col] = LabelEncoder().fit_transform(covariates[col])

  for col in list(covariates.columns):
    if col == "gender" or col == "ethnicity_grouped":
      covariates[col] = covariates[col].astype("category")

  #Spliting the data matrix into numeric and categorical features
  cat_features1 , num_features1 = [], []
  for col in list(covariates.columns):
    if covariates[col].dtypes == "category":
      cat_features1.append(col)
    else:
      num_features1.append(col)

  cat_features = num_features1[:15] +num_features1[17:26]  + cat_features1
  num_features = num_features1[15:17] + num_features1[26:]

  assert len(list(covariates.columns)) == len(cat_features + num_features)

  for col in list(covariates.columns):
    if col in cat_features:
      covariates[col] = covariates[col].astype("category")
  
  # get the embedding sizes for each categorical variable
  embedded_cols = {n: len(col.cat.categories) for n,col in covariates[cat_features].items() if len(col.cat.categories) > 2}
  embedding_sizes = [(n_categories, min(50, (n_categories+1)//2)) for _,n_categories in embedded_cols.items()]

  # Prepare the data matrix as torch tensor
  dfm = pd.concat([covariates.loc[:, cat_features], covariates.loc[:, num_features]], axis = 1)
  dfm1 = df_to_tensor(dfm)

  return (label,label_tensor, label_dict),(surv_time, surv_tensor, surv_time_dict), (x, y), (cat_features, num_features), (embedding_sizes, dfm,dfm1),(df_1, df_2, df_3, df_4, data5)

