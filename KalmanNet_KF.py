"""# **Class: Kalman Filter**
Theoretical Linear Kalman
"""
import torch


class KalmanFilter:

    def __init__(self, SystemModel):
        self.F = SystemModel.F;
        self.F_T = torch.transpose(self.F, 0, 1);
        self.m = SystemModel.m

        self.Q = SystemModel.Q;

        self.H = SystemModel.H;
        self.H_T = torch.transpose(self.H, 0, 1);
        self.n = SystemModel.n

        self.R = SystemModel.R;

        self.T = SystemModel.T;

        # Pre allocate an array for predicted state
        self.x = torch.empty(size=[self.m, self.T])
        self.y_pred = torch.empty((self.n,self.T))
        self.KGs = torch.empty((self.m,self.n,self.T))

        # Predict
        self.F_func = SystemModel.F_function
        self.H_func = SystemModel.H_function

        self.F_Gradient_Function = SystemModel.F_Gradient
        self.H_Gradient_Function = SystemModel.H_Gradient


    def Predict(self):
        # Predict the 1-st moment of x




        self.m1x_prior = self.F_func(self.m1x_posterior)

        self.F = self.F_Gradient_Function(self.m1x_posterior)

        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F, self.m2x_posterior);
        self.m2x_prior = torch.matmul(self.m2x_prior, self.F.T) + self.Q;

        # Predict the 1-st moment of y
        self.m1y = self.H_func(self.m1x_prior);
        self.H  = self.H_Gradient_Function(self.m1x_prior)

        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H, self.m2x_prior);
        self.m2y = torch.matmul(self.m2y, self.H.T) + self.R;

    # Compute the Kalman Gain
    def KGain(self):
        self.KG = torch.matmul(self.m2x_prior, self.H.T)
        self.KG = torch.matmul(self.KG, torch.inverse(self.m2y))
        # self.KG = torch.tensor([[0.3086,0.1586],[0.0644,0.0363]])

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y;

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG, self.dy);

        # Compute the 2-nd posterior moment
        self.m2x_posterior = torch.matmul(self.m2y, torch.transpose(self.KG, 0, 1))
        self.m2x_posterior = self.m2x_prior - torch.matmul(self.KG, self.m2x_posterior)

        # self.m2x_posterior = (torch.eye(self.m)- self.KG.mm(self.H)).mm(self.m2x_prior)

    def Update(self, y):
        self.Predict();
        self.KGain();
        self.Innovation(y);
        self.Correct();

        return self.m1x_posterior;

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = m1x_0
        self.m2x_0 = m2x_0

        #########################

    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y):
        self.m1x_posterior = self.m1x_0
        self.m2x_posterior = self.m2x_0

        for t in range(0, self.T):
            yt = torch.unsqueeze(y[:, t], 1);
            xt = self.Update(yt);
            self.y_pred[:,t] = self.m1y.squeeze()
            self.x[:, t] = torch.squeeze(xt)
            self.KGs[:,:,t] = self.KG