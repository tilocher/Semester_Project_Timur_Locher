import torch
import torch.nn as nn
import numpy as np
import random
from Plot import Plot

from functools import wraps

class Pipeline:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '\\'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt"
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel


    def setModel(self, model):
        self.model = model
        self.model.to(self.model.device)

    def ResetOptimizer(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)


    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay,unsupervised):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch # Number of Samples in Batch
        self.learningRate = learningRate # Learning Rate
        self.weightDecay = weightDecay # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')
        # self.loss_fn = self.max_morm()

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.unsupervised = unsupervised

    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target):

        self.N_E = n_Examples
        self.N_CV = n_CV

        MSE_cv_linear_batch = np.empty([self.N_CV])
        self.MSE_cv_linear_epoch = np.empty([self.N_Epochs])
        self.MSE_cv_dB_epoch = np.empty([self.N_Epochs])

        MSE_train_linear_batch = np.empty([self.N_B])
        self.MSE_train_linear_epoch = np.empty([self.N_Epochs])
        self.MSE_train_dB_epoch = np.empty([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()

            for j in range(0, self.N_CV):
                y_cv = cv_input[j, :, :]
                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_cv = torch.empty(self.ssModel.m, self.ssModel.T)
                y_out_cv = torch.empty(self.ssModel.n, self.ssModel.T)

                for t in range(0, self.ssModel.T):
                    x_out_cv[:, t] = self.model(y_cv[:, t])
                    y_out_cv[:,t] = self.model.m1y.squeeze()

                # Compute Training Loss
                MSE_cv_linear_batch[j] = self.loss_fn(x_out_cv, cv_target[j, :, :]).item() if not self.unsupervised else self.loss_fn(y_out_cv, cv_input[j, :, :]).item()

            # Average
            self.MSE_cv_linear_epoch[ti] = np.mean(MSE_cv_linear_batch)
            self.MSE_cv_dB_epoch[ti] = 10 * np.log10(self.MSE_cv_linear_epoch[ti])

            if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                self.MSE_cv_idx_opt = ti
                torch.save(self.model, self.modelFileName)



            ###############################
            ### Training Sequence Batch ###
            ###############################

            # Training Mode
            self.model.train()

            # Init Hidden State
            self.model.init_hidden()

            Batch_Optimizing_LOSS_sum = 0

            for j in range(0, self.N_B):
                n_e = random.randint(0, self.N_E - 1)

                y_training = train_input[n_e, :, :]
                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_training = torch.empty(self.ssModel.m, self.ssModel.T)
                y_out_training = torch.empty(self.ssModel.n, self.ssModel.T)

                for t in range(0, self.ssModel.T):
                    x_out_training[:, t] = self.model(y_training[:, t])
                    y_out_training[:,t] = self.model.m1y.squeeze()

                # Compute Training Loss
                LOSS = self.loss_fn(x_out_training, train_target[n_e, :, :]) if not self.unsupervised else self.loss_fn(y_out_training, train_input[n_e, :, :])
                MSE_train_linear_batch[j] = LOSS.item()

                Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

            # Average
            self.MSE_train_linear_epoch[ti] = np.mean(MSE_train_linear_batch)
            self.MSE_train_dB_epoch[ti] = 10 * np.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.
            self.optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / self.N_B
            Batch_Optimizing_LOSS_mean.backward()

            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

    def NNTest(self, n_Test, test_input, test_target):
        with torch.no_grad():

            self.N_T = n_Test

            self.MSE_test_linear_arr = np.empty([self.N_T])

            # MSE LOSS Function
            loss_fn = nn.MSELoss(reduction='mean')

            self.model = torch.load(self.modelFileName)

            self.model.eval()

            torch.no_grad()

            for j in range(0, self.N_T):

                y_mdl_tst = test_input[j, :, :]

                self.model.InitSequence(self.ssModel.m1x_0)

                x_out_test = torch.empty(self.ssModel.m, self.ssModel.T)

                for t in range(0, self.ssModel.T):
                    x_out_test[:, t] = self.model(y_mdl_tst[:, t])
                # if j< 15:
                #     Plot_system(x_out_test,test_input[j,:,:],test_target[j,:,:],self.ssModel.Q_sequence[j],self.ssModel.R_sequence[j])
                self.MSE_test_linear_arr[j] = loss_fn(x_out_test, test_target[j, :, :]).item()

            # Average
            self.MSE_test_linear_avg = np.mean(self.MSE_test_linear_arr)
            self.MSE_test_dB_avg = 10 * np.log10(self.MSE_test_linear_avg)

            # Print MSE Cross Validation
            str = self.modelName + "-" + "MSE Test:"
            print(str, self.MSE_test_dB_avg, "[dB]")




    def PlotTrain(self, MSE_KF_linear_arr, MSE_KF_dB_avg):

        self.Plot = Plot(self.folderName, self.modelName)

        self.Plot.NNPlot_epochs(self.N_Epochs, MSE_KF_dB_avg,
                                self.MSE_test_dB_avg, self.MSE_cv_dB_epoch, self.MSE_train_dB_epoch)

        self.Plot.NNPlot_Hist(MSE_KF_linear_arr, self.MSE_test_linear_arr)

