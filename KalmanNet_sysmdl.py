import torch
import numpy as np
from KalmanNet_data import R_decibel_test_ranges
from  tqdm import trange


class SystemModel:

    def __init__(self, F, q, H, r, T,non_linear: bool = False ) :

        ####################
        ### Motion Model ###
        ####################
        self.F = F
        self.m = self.F.size()[0]

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        #########################
        ### Observation Model ###
        #########################
        self.H = H
        self.n = self.H.size()[0]

        self.r = r
        self.R = r * r * torch.eye(self.n)

        ################
        ### Sequence ###
        ################
        # Assign T
        self.T = T

        # Pre allocate an array for current state
        self.x = torch.empty(size=[self.m, self.T])

        # Pre allocate an array for current observation
        self.y = torch.empty(size=[self.n, self.T])

        self.non_linear = non_linear

        if not self.non_linear:
            self.F_function = self.LinearF
            self.H_function = self.LinearH
            self.F_Gradient = self.LinearFgrad
            self.H_Gradient = self.LinearHgrad

    def LinearF(self,x):
        return self.F.mm(x)
    def LinearH(self,x):
        return self.H.mm(x)
    def LinearFgrad(self,x):
        return self.F
    def LinearHgrad(self,x):
        return self.H
    #####################
    ### Init Sequence ###
    #####################
    def InitSequence(self, m1x_0, m2x_0):

        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0


    #########################
    ### Update Covariance ###
    #########################
    def UpdateCovariance_Gain(self, q, r):

        self.q = q
        self.Q = q * q * torch.eye(self.m)

        self.r = r
        self.R = r * r * torch.eye(self.n)

    def UpdateCovariance_Matrix(self, Q, R):

        self.Q = Q

        self.R = R


    ##############################
    ### Generate Sparse Vector ###
    ##############################
    def GenerateSparseVector(self, p, R_gen):

        # Set x0 to be x previous (check the shape ???)
        self.x_prev = self.m1x_0

        # State Evolution
        xt = self.F.matmul(self.x_prev)

        # Input modeled as a process Noise
        P_VEC = torch.zeros(self.m, 1) + torch.transpose(torch.tensor([[0.0, p]]), 0, 1)
        ut = 10 * torch.bernoulli(P_VEC)

        # Additive Process Noise
        xt = xt.add(ut)

        # Emission
        yt = self.H.matmul(xt)

        # Observation Noise
        mean = torch.zeros(self.n)
        er = np.random.multivariate_normal(mean, R_gen, 1)
        er = torch.transpose(torch.tensor(er), 0, 1)

        # Additive Observation Noise
        yt = yt.add(er)

        ########################
        ### Squeeze to Array ###
        ########################
        t = 0

        # Save Current State to Trajectory Array
        self.x[:, t] = torch.squeeze(xt)

        # Save Current Observation to Trajectory Array
        self.y[:, t] = torch.squeeze(yt)

        ################################
        ### Save Current to Previous ###
        ################################
        self.x_prev = xt


    #########################
    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, Q_gen, R_gen):

        # Set x0 to be x previous
        self.x_prev = self.m1x_0

        with torch.no_grad():
            # Generate Sequence Iteratively
            for t in range(0, self.T):
                ########################
                #### State Evolution ###
                ########################

                xt = self.F_function(self.x_prev)

                # Process Noise
                mean = torch.zeros(self.m)
                eq = np.random.multivariate_normal(mean, Q_gen, 1)
                eq = torch.transpose(torch.tensor(eq), 0, 1)
                eq = eq.type(torch.float)

                # Additive Process Noise
                xt = xt.add(eq)

                ################
                ### Emission ###
                ################
                yt = self.H_function(xt)

                # Observation Noise
                mean = torch.zeros(self.n)
                er = np.random.multivariate_normal(mean, R_gen, 1)
                er = torch.transpose(torch.tensor(er), 0, 1)

                # Additive Observation Noise
                yt = yt.add(er)

                ########################
                ### Squeeze to Array ###
                ########################

                # Save Current State to Trajectory Array
                self.x[:, t] = torch.squeeze(xt)

                # Save Current Observation to Trajectory Array
                self.y[:, t] = torch.squeeze(yt)

                ################################
                ### Save Current to Previous ###
                ################################
                self.x_prev = xt

    ######################
    ### Generate Batch ###
    ######################
    def GenerateBatch(self, size,flag):

        # Allocate Empty Array for Input
        self.Input = torch.empty(size, self.n, self.T)

        # Allocate Empty Array for Target
        self.Target = torch.empty(size, self.m, self.T)
        with torch.no_grad():

            ### Generate Examples
            for i in trange(size,desc = '{} Data Gen'.format(flag),position= 0,leave = True):
                # Generate Sequence
                self.GenerateSequence(self.Q, self.R)

                # Training sequence input
                self.Input[i, :, :] = self.y

                # Training sequence output-20
                self.Target[i, :, :] = self.x

    def SetSystemFunctions(self,F_func,H_func):

        self.F_function = F_func
        self.H_function = H_func


    def SetGradientFunctions(self,F_Gradient_function = None,H_Gradient_funciton = None ):

        self.F_Gradient = torch.eye(self.m) if F_Gradient_function == None else F_Gradient_function
        self.H_Gradient = torch.eye(self.n) if H_Gradient_funciton == None else H_Gradient_funciton