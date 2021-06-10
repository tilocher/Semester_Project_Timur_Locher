import torch
import math
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#######################
### Size of DataSet ###
#######################

# Number of Training Examples
N_E = 10000
# N_E = 2000
# N_E = 100
# Number of Cross Validation Examples
N_CV = 100
# N_CV = 100
# N_CV = 2

# Number of Testing Examples
N_T = 10000
# N_T = 2000
# N_T = 2

#################
## Design #10 ###
#################
F10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

H10 = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

############
## 2 x 2 ###
############
m = 2
n = 2
F_design = F10[0:m, 0:m]
H_design = H10[0:n, 10-m:10]
m1x_0_design = torch.ones((m,1))
#m1x_0_design = torch.tensor([[10.0], [-10.0]])
m2x_0_design = 1 * 1 * torch.eye(m)

# Create canonical F
F = torch.eye(m)
F[0,:] = 1

# Create inverse canonical H
H = torch.zeros((n,m))
H[0,:] = 1
for i in range(n):
    for j in range(m):
        if j == m-i-1:
            H[i,j] = 1


T = 80

is_non_linear = False



# F_design = torch.tensor([[1,dt],[0,1]])
# H_design = torch.eye(m)

#############
### 5 x 5 ###
#############
#m = 5
#n = 5
#F_design = F10[0:m, 0:m]
#H_design = H10[0:n, 10-m:10]
#m1x_0_design = torch.zeros(m, 1)
#m1x_0_design = torch.tensor([[1.0], [-1.0], [2.0], [-2.0], [0.0]])
#m2x_0_design = 0 * 0 * torch.eye(m)

# F = F_design
# H  = H_design
m1_0 = m1x_0_design
m2_0 = m2x_0_design

###############################################################################

R_decibel_train_ranges = np.array([-10,-3,0,3,10,20,30])
R_decibel_test_ranges = np.array([-10,-6,-3,0,3,6,10])
QR_ratios = np.array([-20,-10,0])

def DataGen(SysModel_data, fileName):

    ##################################
    ### Generate Training Sequence ###
    ##################################
    SysModel_data.GenerateBatch(N_E,'Training')
    training_input = SysModel_data.Input
    training_target = SysModel_data.Target

    ####################################
    ### Generate Validation Sequence ###
    ####################################
    SysModel_data.GenerateBatch(N_CV,'Cross Val')
    cv_input = SysModel_data.Input
    cv_target = SysModel_data.Target

    ##############################
    ### Generate Test Sequence ###
    ##############################
    SysModel_data.GenerateBatch(N_T,'Testing')
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target

    #################
    ### Save Data ###
    #################
    torch.save([training_input, training_target, cv_input, cv_target, test_input, test_target], fileName)

def DataLoader(fileName):
    [training_input, training_target, cv_input, cv_target, test_input, test_target] = torch.utils.data.DataLoader(torch.load(fileName),pin_memory = True)
    training_input = training_input.squeeze()
    training_target = training_target.squeeze()
    cv_input = cv_input.squeeze()
    cv_target =cv_target.squeeze()
    test_input = test_input.squeeze()
    test_target = test_target.squeeze()

    return [training_input, training_target, cv_input, cv_target, test_input, test_target]

def TestDataGen(SysModel_data, fileName):
    SysModel_data.GenerateBatch(N_T,'Test')
    test_input = SysModel_data.Input
    test_target = SysModel_data.Target
    torch.save([test_input, test_target], fileName)

def TestDataLoad(fileName):
    [test_input, test_target] = torch.utils.data.DataLoader(torch.load(fileName), pin_memory=True)
    test_input = test_input.squeeze()
    test_target = test_target.squeeze()
    return [test_input,test_target]



# Simple Non-linear:

###################
### Read first ####
###################
# When constructing the gradeints of the functions, they should be able, to output
# the correct matrix for multiple inputs. So if we want to know the gradients for
# k-states, the output should be a (m x m x k) tensor in the case of f!

def F_grad(x: torch.Tensor):

    # Parameters
    alpha = 0.9
    beta = 1.1
    phi = np.pi/10
    delta = 0.01

    # Dimensionality of the states
    m = x.shape[0]

    # Number of inputs
    number_input = x.shape[-1]

    # initialize output tensor
    out = torch.zeros((m,m,number_input))


    # Calculate the gradient based on the non-linear function
    for j in range(number_input):
        mul = beta * alpha * torch.cos(beta * x[:,j] + phi)

        for i in range(m):
            out[i,i,j] = mul[i]

    return out.squeeze()

def H_grad(x: torch.Tensor):

    # Parameters
    a = 1.0
    b = 1.0
    c = 0.0

    # Dimesnion of the state
    n = x.shape[0]
    m = x.shape[1]

    # Number of inputs
    num = x.shape[-1]

    # Initialize output
    out = torch.zeros((n, m, num))

    # Calculate function
    for p in range(num):
        mul = 2 * a * (b * x[:,p] + c) * b

        for i in range(n):
            out[i,i,p] = mul[i]

    return out.squeeze()

def F_func(x: torch.Tensor):
    alpha = 0.9
    beta = 1.1
    phi = np.pi/10
    delta = 0.01

    return alpha * torch.sin(beta*x + phi) + delta

def H_func(x: torch.Tensor):
    a = 1.0
    b = 1.0
    c = 0.0

    return a*(b*x+c)**2