
import torch
import torch.nn as nn
import torch.nn.functional as func
from KalmanNet_nn import KalmanNetNN
import numpy as np
import random
from Plot import Plot
from Pipeline import Pipeline



class KalmanNetTestNN(KalmanNetNN):


    ###################
    ### Constructor ###
    ###################

    def __init__(self,):

        super().__init__()

        self.__dict__ = KalmanNet.__dict__
        self.N_Epochs = number_of_epochs
        self.learning_rate = learningRate
        self.weight_decay = weightDecay
        self.optimizer = torch.optim.Adam(self.parameters(),lr = self.learning_rate, weight_decay= self.weight_decay)
        self.modelName = modelName
        self.folderName = folderName
        self.unsupervised = True
        self.batch_size_test  = batch_size
