import torch
import numpy as np
from KalmanNet_sysmdl import SystemModel
from KalmanNet_data import N_E,N_CV,N_T
from matplotlib import pyplot as plt
from tqdm import trange
from KalmanNet_KF import KalmanFilter


class Least_squares_estimator():

    # Initialize the Filter with the system dynamics
    def __init__(self,System: SystemModel,model_name: str):

        # Initialize System parameters
        self.System = System

        self.F = System.F
        self.H = System.H
        self.T = System.T

        self.m = System.m
        self.n = System.n

        self.F_function = System.F_function
        self.H_function = System.H_function
        self.F_gradient = System.F_Gradient
        self.H_gradient = System.H_Gradient

        # First posterior
        self.m1x_0 = System.m1x_0

        # Check if the system is non-linear
        self.non_linear = System.non_linear

        # Preallocate Estimates
        self.x_est = torch.empty((self.m, self.T))
        self.y_est = torch.empty((self.n, self.T))
        self.Innovations = torch.empty((self.n,self.T))
        self.priors = torch.empty((self.m,self.T))
        self.next_prediction = torch.empty((self.n,self.T))

        # Parameters for plotting
        self.wrong_KF_MSE_is_current = self.KF_MSE_is_current = False
        self.PLotLines = np.array([])
        self.PLotLineLabels = np.array([])

        # Model name
        self.name = model_name

    ##############################################################################################

    # Initialize the first posterior
    def InitializeSequence(self, m1x0: torch.Tensor):
        self.m1x_posterior = m1x0

    ##############################################################################################

    # Initialize the filter settings
    def InitializeFilter(self,unsupervised: bool = False, offline: bool = True):

        self.unsupervised = unsupervised
        self.offline = offline

    ##############################################################################################

    # Initialize the offline training parameters
    def InitializeOfflineTrainingParameters(self, learning_rate: float = 1,batch_size: int = 50, epochs: float = 20,
                                      two_point_step: bool = True):

        # Learning rate of the filter
        self.lr = learning_rate

        # Batch size for training
        self.batch_size = batch_size

        # Number of training epochs
        self.epochs = epochs

        # Step size method for optimization, recommended True for supervised and False for unsupervised
        self.two_point_step = two_point_step

        # Loss function for the filter
        self.loss_fn = torch.nn.MSELoss(reduction= 'mean')

    ##############################################################################################

    # Initialize the offline training parameters
    def InitializeOnlineTrainingParameters(self, learning_rate: float = 1,
                                            window_size: int = 10, two_point_step: bool = True,
                                            online_split_perc: int = 50,
                                            stride: int = 1):
        # Learning rate of the filter
        self.lr = learning_rate

        # Window size for online learning
        self.window_size = window_size

        # Step size method for optimization, recommended True for supervised and False for unsupervised
        self.two_point_step = two_point_step

        # Loss function for the filter
        self.loss_fn = torch.nn.MSELoss(reduction='mean')

        # Split for testing in supervised online training
        self.training_split = 100 if self.unsupervised else online_split_perc

        # Stride of the window
        self.stride = stride

    ##############################################################################################

    # Prediction step of KF
    def predict(self):

        # Calculate prior based on current information
        self.m1x_prior = self.F_function(self.m1x_posterior)

        # Calculate observation prediction
        self.d1y = self.H_function(self.m1x_prior)

    ##############################################################################################

    # Update step of KF
    def Update(self,y: torch.Tensor):

        # Ensure shape
        y = y.reshape(self.n,1)

        # Calculate Innovation
        self.Innovation = y - self.d1y

        # Calculate posterior
        self.m1x_posterior = self.m1x_prior + self.KG.mm(self.Innovation)

    ##############################################################################################

    # Create KF output from the current best KG, or a given KG
    def Generate_Sequence(self, test_input: torch.Tensor):

        # Reset posterior
        self.InitializeSequence(self.m1x_0)

        # Calculate estimates over a trajectory
        for i in range(self.T):

            # Prior step
            self.predict()

            # Append to priors
            self.priors[:,i] = self.m1x_prior.squeeze()

            # Observation prediction
            self.y_est[:, i] = self.d1y.squeeze()

            # Update prediction
            self.Update(test_input[:,i])

            # Innovations
            self.Innovations[:,i] = self.Innovation.squeeze()

            # Prediction if KG = 0, for unsupervised loss
            self.next_prediction[:,i] = self.H_function(self.F_function(self.m1x_prior)).squeeze()

            # State prediction
            self.x_est[:, i] = self.m1x_posterior.squeeze()

        return self.x_est

    ##############################################################################################

    # Calculate loss based on the filter setting
    def CalculateLoss(self, input_data_set: torch.Tensor, target_data_set: torch.Tensor, ne: int):

        # Unsupervised loss
        if self.unsupervised:
            inputs = input_data_set[ne]
            loss = self.loss_fn(self.y_est, inputs)

        # Supervised loss
        else:
            target = target_data_set[ne]
            loss = self.loss_fn(self.x_est, target)
        return loss

    ##############################################################################################

    # Build the LS-algo input and target
    def BuildLS(self,ne: int,time: int = -1):

        window = min(self.window_size,time) if not self.offline else 0
        first_time_step = 0 if self.offline else time-window
        final_time_step = self.T if self.offline else time+1

        if self.unsupervised:
            observations = self.training_input[ne,:,first_time_step:final_time_step]
            self.LS_input = self.Innovations[:,first_time_step:final_time_step-1]
            self.LS_target = observations[:,1:] - self.next_prediction[:,first_time_step:final_time_step-1]

        else:
            targets = self.training_target[ne,:,first_time_step:final_time_step]
            self.LS_input = self.Innovations[:,first_time_step:final_time_step]
            self.LS_target = targets - self.priors[:,first_time_step:final_time_step]

    ##############################################################################################

    # Trajectory step calculation
    def CalculateTrajectoryStep(self):

        if self.unsupervised:
            # Calculate the optimization step for a linear system in an unsupervised use case
            if not self.non_linear:
                dK = self.LS_target
                dK = dK - self.H.mm(self.F).mm(self.KG).mm(self.LS_input)
                dK = -(self.H.mm(self.F)).T.mm(dK)
                dK = dK.mm(self.LS_input.T)

            # Non-linear unsupervised loss
            else:

                F_gradient_evaluated = self.F_gradient(self.priors[:,:-1] + self.KG.mm(self.LS_input))
                H_gradient_evaluated = self.H_gradient(self.F_function(self.priors[:,:-1] + self.KG.mm(self.LS_input)))


                # Convert to numpy, since torch does not currently support tensor products
                F_gradient_evaluated = F_gradient_evaluated.T.numpy()
                H_gradient_evaluated = H_gradient_evaluated.T.numpy()
                self.LS_target = self.LS_target.numpy()


                x1 = np.tensordot(self.LS_target,H_gradient_evaluated,axes=([-1],[0]))
                x2 = np.tensordot(F_gradient_evaluated,x1.T,axes= ([-1,-2],[0,1]))
                x2 = torch.from_numpy(x2)
                # dK = -self.LS_input.mm(x2)
                dK = - x2.T.mm(self.LS_input.T)

        else:
            # Calculate the optimization step for a linear system in an supervised use case, both linear and non-linear
            dK = -(self.LS_target - self.KG.mm(self.LS_input)).mm(self.LS_input.T)

        # Normalize step
        dK = dK/self.LS_target.shape[-1]

        return dK

    ##############################################################################################

    # Calculate the batch optimization step based on the setting
    def Step(self):

        # Normalize step
        self.dK = self.dK / self.batch_size if self.offline else self.dK

        # Use two-point-step method as described at page 16 of the report
        if self.two_point_step:
            alpha = self.lr * torch.norm((self.KG - self.KG_prev).T.mm(self.dK - self.dK_prev)) / (
                torch.norm(self.dK - self.dK_prev) ** 2) if not torch.norm(self.KG_prev) == 0 else self.lr / torch.norm(self.dK)

        # Use a basic normalized step
        else:
            alpha = self.lr / torch.norm(self.dK)

        # Update previous parameters
        self.KG_prev = self.KG
        self.dK_prev = self.dK

        # Update KG
        self.KG = self.KG - alpha * self.dK

        return self.KG

    ##############################################################################################

    # Select the correct training function, based on the filter settings
    def Estimate_KalmanGain(self,training_input: torch.Tensor = None, training_target: torch.Tensor = None,
                            cv_input: torch.Tensor = None, cv_target: torch.Tensor = None):

        # To avoid unnecessary tracking of the gradient from pytorch and speed up computation
        with torch.no_grad():
            if self.offline:
                return self.Estimate_KG_offline(training_input,training_target,cv_input,cv_target)
            else:
                return self.Estimate_KG_online(training_input,training_target)

    ##############################################################################################

    # Estimate the KG on a offline data set
    def Estimate_KG_offline(self,training_input: torch.Tensor,training_target: torch.Tensor,
                            cv_input: torch.Tensor,cv_target: torch.Tensor):

       #Initialize data-sets

        self.training_input = training_input
        self.training_target = training_target
        self.cv_input = cv_input
        self.cv_target = cv_target

        # Initialize losses
        self.train_loss = torch.empty((self.epochs))
        self.cv_loss = torch.empty((self.epochs))

        # Initialize the KG
        self.KG = torch.zeros((self.m, self.n))

        # Initialize optimization step
        self.dK = torch.zeros((self.m,self.n))

        # Initialize previous parameters for the step size
        self.KG_prev = torch.zeros((self.m, self.n))
        self.dK_prev = torch.zeros((self.m, self.n))

        # First CV loss and best KG
        self.best_cv_loss = torch.tensor([10e6])
        self.Best_KG = self.KG
        self.Best_idx = 0

        # Start the training
        for nepoch in range(self.epochs):

            print('Begin Epoch:',nepoch,'/',self.epochs)

            # Begin CV
            cv_loss = 0

            for i in range(N_CV):

                # Calculate KF filter output
                self.Generate_Sequence(cv_input[i])

                # Calculate CV trajectory loss
                loss = self.CalculateLoss(cv_input,cv_target,i)

                # Update total CV loss
                cv_loss += loss

            # Calc. CV batch loss in dB
            cv_batch_loss = 10 * np.log10(cv_loss / N_CV)

            # Check if KG is an improvement
            if cv_batch_loss < self.best_cv_loss and nepoch!=0:
                self.Best_KG = self.KG
                self.best_cv_loss = cv_batch_loss
                self.Best_idx = nepoch

            # Update epoch CV loss
            self.cv_loss[nepoch] = cv_batch_loss

            # Initialize new opt. step
            self.dK  = torch.zeros((self.m,self.n))
            # Begin Training
            training_loss = 0
            for nbatch in range(self.batch_size):

                # Random sample
                ne = np.random.randint(0,N_E)

                # Generate KF output
                self.Generate_Sequence(self.training_input[ne])

                # Build the input and target for the LS optimization
                self.BuildLS(ne)

                # Calculate the optimization step for the current trajectory
                dK = self.CalculateTrajectoryStep()

                # Update the batch optimization step
                self.dK = self.dK + dK

                # Loss From
                loss = self.CalculateLoss(self.training_input,self.training_target,ne)

                training_loss += loss

            # Take optimization step
            self.Step()

            # Track training statistics
            epoch_loss = training_loss / self.batch_size
            epoch_loss = 10* np.log10(epoch_loss)
            self.train_loss[nepoch] = epoch_loss

           # Print statistics
            print('CV Batch loss -', cv_batch_loss.item(), '[dB]')
            print('Best CV loss -',self.best_cv_loss.item(),'[dB] at Index:',self.Best_idx)
            print('Training loss -',epoch_loss.item(),'[dB]','\n')


        self.KG  = self.Best_KG
        return self.KG

    ##############################################################################################

    # Train online for a fraction of a given trajectory and test it on the rest
    def Estimate_KG_online(self,training_input,training_target):

        nTrajectories = training_target.shape[0]

        # Initialize losses
        self.dataset_loss = torch.empty((nTrajectories,1))
        self.TrajectoryLosses = torch.empty((nTrajectories,self.T))


        # Split into training and testing set
        split = int(self.training_split/100*self.T)

        self.KG = torch.zeros((self.m,self.n))

        # Initialize previous parameters for the step size
        self.KG_prev = torch.zeros((self.m, self.n))
        self.dK_prev = torch.zeros((self.m, self.n))

        self.training_input = training_input
        self.training_target = training_target

        # If we use an unsupevised scheme, we need to skip the first observation and posterior
        start = int(self.unsupervised)

        for n in trange(nTrajectories,desc = 'Online Learning',position = 0,leave = True):

            # Initialize Sequence
            self.InitializeSequence(self.m1x_0)

            # Start going through a trajectory
            for t in range(self.T):

                #Prediction of the KF
                self.predict()

                # Append to priors
                self.priors[:, t] = self.m1x_prior.squeeze()

                # Observation prediction
                self.y_est[:, t] = self.d1y.squeeze()

                # Observation prediction
                self.y_est[:, t] = self.d1y.squeeze()

                # Update based on prior and KG
                self.Update(self.training_input[n,:,t])

                # Innovations
                self.Innovations[:, t] = self.Innovation.squeeze()

                # Append the next prediction, assuming KG = 0 for the loss
                self.next_prediction[:,t] = self.H_function(self.F_function(self.m1x_prior)).squeeze()

                # State prediction
                self.x_est[:, t] = self.m1x_posterior.squeeze()


                # Stop optimizing, at the given stoping point in the supervised case. Also only update every 'stride'
                # time steps
                if t <= split and t>=start and t%self.stride==0:

                    # Initialize optimization step
                    self.dK = torch.zeros((self.m,self.n))

                    # Build LS input and target
                    self.BuildLS(n,t)

                    # Calculate the optimization step
                    self.dK = self.CalculateTrajectoryStep()

                    # Take optimizatio step
                    self.Step()

                # Calculate loss
                trajectory_loss = self.loss_fn(self.x_est[:,t],self.training_target[n,:,t]).item()
                self.TrajectoryLosses[n, t] = trajectory_loss


            # Calculate mean trajectory loss, after one window optimization
            self.dataset_loss[n] = torch.mean(self.TrajectoryLosses[n,self.window_size:])

        # Remove outliners. The algorithm can diverge in about 0.01% of the time, skewing the mean.
        # The precentage of outliners is calcualted
        x = self.dataset_loss.numpy()
        self.dataset_loss_converged  = x[abs(x-np.mean(x)) < 1* np.std(x)]
        self.dataset_loss_converged = torch.from_numpy(self.dataset_loss_converged)
        self.average_converged_loss = 10* np.log10(torch.mean(self.dataset_loss_converged).item())
        print('Online Loss: ',self.average_converged_loss,'[dB]',', ratio converged:',
              self.dataset_loss_converged.shape[0]/self.dataset_loss.shape[0])

    ##############################################################################################

    # PLot training and CV
    def PlotTraining(self):

        start_plot = 0
        t = torch.linspace(0, self.epochs, self.epochs)

        # Plot training and CV loss
        plt.plot(t[start_plot:], self.train_loss[start_plot:], 'or', label='Train Loss', linewidth=1, markersize=3)
        plt.plot(t[start_plot:], self.cv_loss[start_plot:], 'ob', label='CV Loss', linewidth=1, markersize=3)


        for n,value in enumerate(self.PLotLines):
            y = torch.ones((self.epochs))*value
            plt.plot(t,y,'--',label = self.PLotLineLabels[n])

        # Show plot
        plt.grid()
        plt.legend()
        plt.xlabel('Training Itteration')
        plt.ylabel('Training Loss [dB]')
        title = 'Unsupervised Training Loss' if self.unsupervised else 'Supervised Training Loss'
        q = round(10*np.log10(self.System.q),3)
        r = round(10*np.log10(self.System.r),3)
        title = title + ', Batch Size: {}'.format(str(self.batch_size)) + ', q = {}dB, r = {}dB'.format(q,r)
        plt.title(title)
        plt.savefig('LS\\convergence_plot.eps')
        plt.show()

    ##############################################################################################

    # KF filter MSE for plotting
    def SetKalmanFilterMSE(self,KF_MSE):
        self.KF_MSE = KF_MSE
        self.KF_MSE_is_current = True

    ##############################################################################################

    # Wrong KF filter MSE for plotting
    def SetWrongKalmanFilterMSE(self,KF_MSE):
        self.wrong_KF_MSE = KF_MSE
        self.wrong_KF_MSE_is_current = True

    ##############################################################################################

    # Save LSE object
    def save(self):

        torch.save(self,self.name)

    ##############################################################################################

    # Set and Reset Plot lines for refrence
    def SetPlotLines(self,value: torch.Tensor, label: str):
        self.PLotLines = np.append(self.PLotLines,value.item())
        self.PLotLineLabels = np.append(self.PLotLineLabels,label)
    def ResetPlotLines(self):
        self.PLotLines = np.array([])
        self.PLotLineLabels = np.array([])


















