from KalmanNet_data import F,m1_0,m2_0,H,R_decibel_train_ranges,QR_ratios,T,m,n, N_T,is_non_linear
import torch
from KalmanNet_data import DataGen,DataLoader
from KalmanNet_sysmdl import SystemModel
from tqdm import tqdm,trange
import numpy as np
from  KalmanFilter_test import KFTest
from KalmanNet_LMSE_Test import LMSETest
from matplotlib import pyplot as plt
from Least_squares_estimator import Least_squares_estimator
from KalmanNet_data import F_func,H_func,F_grad,H_grad
from KalmanNet_data import R_decibel_train_ranges,QR_ratios


# Build initial System
r = q =1
ssSystem = SystemModel(F,q,H,r,T,is_non_linear)
ssSystem.InitSequence(m1_0,m2_0)
# If non Linear:
# ssSystem.SetSystemFunctions(F_func,H_func)
# ssSystem.SetGradientFunctions(F_grad,H_grad)


for nration, ratio in enumerate(QR_ratios):

    print('Training ratio:',ratio,'[dB]')

    for nRdB, rdB in enumerate(R_decibel_train_ranges):


        print('Training observation noise 1/R^2:',rdB,'[dB]')


        data_file_name = 'Data\\Ratio_{}_R_{}.pt'.format(ratio,rdB)
        model_name = 'LS\\Ratio_{}_R_{}.pt'.format(ratio,rdB)


        r = 10**(-rdB/20)
        q = 10**((ratio-rdB)/20)

        ssSystem.UpdateCovariance_Gain(q,r)

        DataGen(ssSystem,data_file_name)
        [train_input,train_target,cv_input,cv_target,test_input,test_target] = DataLoader(data_file_name)

        _,_,states,obs = KFTest(ssSystem,test_input,test_target)

        LSE = Least_squares_estimator(ssSystem,model_name)
        LSE.InitializeFilter(unsupervised= False, offline= True)
        LSE.InitializeOfflineTrainingParameters(learning_rate=2, two_point_step=True, batch_size= 100,epochs=30)
        # LSE.InitializeOnlineTrainingParameters(learning_rate=1,window_size=20,two_point_step=False,stride=3)
        LSE.Estimate_KalmanGain(train_input,train_target,cv_input,cv_target)
        LSE.SetPlotLines(states,'KF Loss')
        LSE.PlotTraining()
        LSE.save()

        LMSETest(LSE,test_input,test_target)