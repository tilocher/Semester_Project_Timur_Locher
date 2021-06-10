import torch
from Least_squares_estimator import Least_squares_estimator
from KalmanNet_data import N_T
from tqdm import trange


def LMSETest(LSE: Least_squares_estimator,Test_Input: torch.Tensor,Test_Target: torch.Tensor):

    # Loss
    loss_fn = torch.nn.MSELoss(reduction= 'mean')


    # Preallocate Linear Loss
    MSE_LMSE_linear_arr = torch.empty(N_T)

    # Initialize LMSE Object


    for i in trange(N_T,desc = 'LMSE Loss Run',position= 0,leave = True):

        # Estimate Sequence from SS Kalman Gain
        LSE.Generate_Sequence(Test_Input[i])


        # Calculate loss
        MSE_LMSE_linear_arr[i] = loss_fn(LSE.x_est,Test_Target[i,:,:])

    MSE_LMSE_linear_avg = torch.mean(MSE_LMSE_linear_arr)
    MSE_LMSE_dB_avg = 10 * torch.log10(MSE_LMSE_linear_avg).item()

    print('Least Squares - MSE loss: ',MSE_LMSE_dB_avg,'[dB]')


    return [MSE_LMSE_linear_arr.numpy(),MSE_LMSE_linear_avg.item(),MSE_LMSE_dB_avg]






