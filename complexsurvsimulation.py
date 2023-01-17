# -*- coding: utf-8 -*-
"""ComplexSurvSimulation

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w_i-ssYxto5H5jj1C9UuVsdccrxWaZIa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from scipy.special import expit



def __cmpxRec__():

  """
   @Author: Martin Pius
  --------------------
  This method generate recurrent events with competing risks survival data

  Arguments
  -----------
  rec_l: List of int, Total number of samples at each recurrent step

  returns:
  --------
  rec1_t: np.nd-array, risk_1 RECURENT times
  rec2_t: np.nd-array, risk_2 RECURRENT times
  cov_list: np.nd-arrY, COVARIATES
  """
  np.random.seed(111)
  N = 30000
  mean = [0,0,0,0]
  cov = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
  beta_1 = np.array([1,1,1,1])
  beta_2 = np.array([1,1,1,1])
  beta_3 = np.array([1,1,1,1])
  beta_4 = np.array([1,1,1,1])
  omega_1, omega_2, omega_3 = np.array([10,10,10,10]), np.array([10,10,10,10]), np.array([10,10,10,10])
  lambda_1, lambda_2 = np.array([1,1,1,1]), np.array([1,1,1,1])

  t_rec1 = []
  np.random.seed(120)
  C1 = np.random.uniform(low = 1, high = 31)
  for i in range(30000):
    j = 0
    while j < 5:
      T1_prv = 0
      X_1 = np.random.multivariate_normal(mean, cov, 1)
      X_2 = np.random.multivariate_normal(mean, cov, 1)
      X_3 = np.random.multivariate_normal(mean, cov, 1)
      alpha_1 = omega_1.T * np.abs(X_1) + (omega_3.T * np.abs(X_3))**2
      alpha_2 = omega_2.T * np.abs(X_2) + (omega_3.T * np.abs(X_3))**2
      u_1 = np.random.uniform(size = (4,1))
      u_2 = np.random.uniform(size = (4, 1))

      q1 = alpha_1.T * np.log(u_1)
      a1 = beta_1*X_1 + beta_2*X_2 + beta_3*X_3 
      q2 = np.exp(a1)
      q3 = (q1/q2.T)
      b0 = 1 - q3
      b1 = np.log(b0)

      w1 = b1.T//lambda_1
      w1 = w1.sum(axis = 1)
    
      T1_nxt = T1_prv + w1 
      if T1_nxt > np.array([30]):
        T1_nxt = np.array([30])
      else:
        T1_prv = T1_nxt
    
        j+=1
      t_rec1.append(T1_nxt)

  t_rec2 = []
  cov_list = []
  np.random.seed(120)
  C2 = np.random.uniform(low = 1, high = 31)
  for i in range(30000):
    j = 0
    while j < 5:
      T2_prv = 0
      X_1 = np.random.multivariate_normal(mean, cov, 1)
      X_2 = np.random.multivariate_normal(mean, cov, 1)
      X_3 = np.random.multivariate_normal(mean, cov, 1)
      alpha_1 = omega_1.T * np.abs(X_1) + (omega_3.T * np.abs(X_3))**2
      alpha_2 = omega_2.T * np.abs(X_2) + (omega_3.T * np.abs(X_3))**2
      u_1 = np.random.uniform(size = (4,1))
      u_2 = np.random.uniform(size = (4, 1))
      q1 = alpha_1.T * np.log(u_1)
      a1 = beta_1*X_1 + beta_2*X_2 + beta_3*X_3 
      q2 = np.exp(a1)
      q3 = (q1/q2.T)
      b0 = 1 - q3
      b1 = np.log(b0)

      p1 = alpha_2.T * np.log(u_2)
      a1 = beta_1*X_1 + beta_2*X_2 + beta_3*X_3 
      p2 = np.exp(a1)
      p3 = (p1/p2.T)
      b0 = 1 - p3
      b1 = np.log(b0)
      w2 = b1.T//lambda_2
      w2 = w2.sum(axis = 1)

      T2_nxt = T2_prv + w2 
      if T2_nxt > np.array([30]):
        T2_nxt = np.array([30])
      else:
        T2_prv = T2_nxt
    
        j+=1
      t_rec2.append(T2_nxt)
      cov_list.append([X_1, X_2,X_3])
  return t_rec1, t_rec2, cov_list



def __CreateCmpxRec__():
  """
  @Author: Martin Pius
  ---------------------
  This method use __cmpxRec__ module to create the
  final complex survival data.

  return
  -------
  pd.DataFrame
  """
  print("This may take a while!.........please wait when we generate the requested data.......")
  t_rec1, t_rec2, cov_list = __cmpxRec__()
  t1 = np.array(t_rec1)
  t2 = np.array(t_rec2)
  x = np.array(cov_list)
  x1 = x.reshape(x.shape[0], -1)

  t1 = t1[:150000]
  t2 = t2[:150000]
  x1 = x1[:150000, :]
  t1 = pd.DataFrame(t1.flatten(), columns = ['t1'])
  t2 = pd.DataFrame(t2.flatten(), columns = ['t2'])
  dt = pd.DataFrame(x1, columns = ['x11','x12','x13','x14','x21','x22','x23','x24','x31','x32','x33','x34'])
  subject_id = np.arange(1, 30001)
  subject_id = subject_id.repeat(5)

  df_r = pd.concat([t1,t2,dt], axis = 1)
  df_r['subject_id'] = pd.Series(subject_id)
  df_rt = df_r.set_index(subject_id)
  cv_r = pd.concat([df_rt.iloc[:,-1],df_rt.iloc[:,:-1]], axis = 1)

  rec_time = []
  for i in range(len(cv_r)):
    if cv_r.iloc[i,1] < cv_r.iloc[i,2]:
      rec_time.append(cv_r.iloc[i,1])
    else:
      rec_time.append(cv_r.iloc[i,2])

  rec_time = pd.Series(rec_time)
  df = pd.concat([pd.DataFrame(rec_time, columns = ["recurent_time"]),cv_r.reset_index()], axis = 1)
  df.drop(['index'], axis = 1, inplace = True)
  df_ = df.iloc[:, [1,2,3,0,4,5,6,7,8,9,10,11,12,13,14,15]]

  c = np.random.uniform(low = 1, high = min(max(cv_r.iloc[:,1]), max(cv_r.iloc[:,2])), size = int(0.45 *len(cv_r)))
  c1 = np.random.uniform(low = 1, high = max(cv_r.iloc[:,1]), size = int(0.45 *len(cv_r)))
  c2 = np.random.uniform(low = 1, high = max(cv_r.iloc[:,2]), size = int(0.45 *len(cv_r)))
  z1 = np.zeros(shape = (len(cv_r)-len(c),1))

  C = np.concatenate([c.reshape(c.shape[0],1), z1])
  C1 = np.concatenate([c1.reshape(c1.shape[0],1), z1])
  C2 = np.concatenate([c2.reshape(c2.shape[0],1), z1])

  C, C1, C2 = C.flatten(), C1.flatten(), C2.flatten()
  np.random.shuffle(C)
  np.random.shuffle(C1)
  np.random.shuffle(C2)

  Cns = pd.DataFrame(data = {"C": C, "C1": C1, "C2": C2})
  my_recData = pd.concat([Cns,df_], axis = 1)
  rec_Data = my_recData.iloc[:, [3,4,5,1,2,0,6,7,8,9,10,11,12,13,14,15,16,17,18]]

  rec_Data[['t1']] = rec_Data[['t1']].astype(int)
  rec_Data[['t2']] = rec_Data[['t2']].astype(int)
  rec_Data[['C']] = rec_Data[['C']].astype(int)
  rec_Data[['C1']] = rec_Data[['C1']].astype(int)
  rec_Data[['C2']] = rec_Data[['C2']].astype(int)
  rec_Data[['recurent_time']] = rec_Data[['recurent_time']].astype(int)

  I = []
  for i in range(len(rec_Data)):
    if (rec_Data.iloc[i, 5]==0) and (rec_Data.iloc[i,1] < rec_Data.iloc[i,2]):
      I.append(1)
    elif (rec_Data.iloc[i, 5]==0) and (rec_Data.iloc[i,1] > rec_Data.iloc[i,2]):
      I.append(2)
    else:
      I.append(0)

  I = pd.Series(I)
  rec_Data["label"] = I

  CmpxData_rec  = rec_Data.iloc[:, [0,1,2,3,4,5,6,-1,7,8,9,10,11,12,13,14,15,16,17,18]]

  dt_r1 = CmpxData_rec.iloc[0:30000,:]
  dt_r2 = CmpxData_rec.iloc[30000:60000,:]
  dt_r3 = CmpxData_rec.iloc[60000:90000,:]
  dt_r4 = CmpxData_rec.iloc[90000:120000,:]
  dt_r5 = CmpxData_rec.iloc[120000:,:]
  rec_dfms = [dt_r1, dt_r2, dt_r3, dt_r4, dt_r5]

  events_data = pd.DataFrame(data = 
                {"stamps": ["T1", "T2","T3","T4","T5"],
                "Risk1":[list(Counter(rec_dfms[i].C1==0).values())[0] for i in range(len(rec_dfms))],
                "Censored_Risk1": [list(Counter(rec_dfms[i].C1==0).values())[1] for i in range(len(rec_dfms))],
                "Risk2":[list(Counter(rec_dfms[i].C2==0).values())[0] for i in range(len(rec_dfms))],
                "Censored_Risk2": [list(Counter(rec_dfms[i].C2==0).values())[1] for i in range(len(rec_dfms))]}) 


  return CmpxData_rec, events_data, rec_dfms

