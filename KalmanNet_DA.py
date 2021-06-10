from KalmanNet_data import F,m1_0,m2_0,H,R_decibel_train_ranges,QR_ratios,T,m,n, N_T,N_CV,N_E
import torch
from KalmanNet_data import DataGen,DataLoader
from KalmanNet_sysmdl import SystemModel

from  KalmanFilter_test import KFTest

from KalmanNet_nn import KalmanNetNN
from Pipeline import Pipeline
import datetime


now = datetime.datetime.now()
KNet_folder = 'KNet'
KNet_model_name = 'KNet_model'

# Set first training point
r_decibel = -10
ratio_decibel = -20

# Convert to non-dB
r_non_dB = 10**(r_decibel/20)
q_non_dB = 10**((ratio_decibel+r_decibel)/20)

# Initialize SS-model
System = SystemModel(F,q_non_dB,H,r_non_dB,T,non_linear=False)
System.InitSequence(m1_0,m2_0)

# Generate Data
# DataGen(System,'Data\\Initial_data.pt')
[train_input,train_target,cv_input,cv_target,test_input,test_target] = DataLoader('Data\\Initial_data.pt')

# Base line KF test
KFTest(System,test_input,test_target)

# Initialize Pipeline
KNet_Pipeline = Pipeline(now,KNet_folder,KNet_model_name)
KNet_Pipeline.setssModel(System)
KNET_NN = KalmanNetNN()
KNET_NN.Build(System)
KNet_Pipeline.setModel(KNET_NN)
KNet_Pipeline.setTrainingParams(n_Epochs=50,n_Batch=50,learningRate=1e-3,weightDecay=1e-6,unsupervised=False)

# First training
KNet_Pipeline.NNTrain(N_E,train_input,train_target,N_CV,cv_input,cv_target)

# First evaluation
KNet_Pipeline.NNTest(N_T,test_input,test_target)

# Set the changed covariance
r_changed_dB = r_decibel + 10
r_non_dB_changed = 10**(r_changed_dB/20)
System.UpdateCovariance_Gain(q_non_dB,r_non_dB_changed)
System.InitSequence(m1_0,m2_0)

# Generate changed dataset
DataGen(System,'Data\\data_changed.pt')
[train_input_changed,train_target_changed,cv_input_changed,
 cv_target_changed,test_input_changed,test_target_changed] = DataLoader('Data\\data_changed.pt')

# Evaluate baseline for second Dataset
KFTest(System,test_input_changed,test_target_changed)

# Evaluate the unchanged performance
KNet_Pipeline.NNTest(N_T,test_input_changed,test_target_changed)

# Begin DA
# KNet_Pipeline.setUnsupervised(True)

# Reset optimizer
KNet_Pipeline.ResetOptimizer()

# Re-train
percentage = 0.15
KNet_Pipeline.setTrainingParams(n_Epochs=15,n_Batch=40,learningRate=1.5e-3,weightDecay=5e-6,unsupervised=True)
KNet_Pipeline.NNTrain(int(N_E*percentage),train_input_changed,None,int(N_CV*percentage),cv_input_changed,None)

# Evaluate retrained performance
KNet_Pipeline.NNTest(N_T,test_input_changed,test_target_changed)

