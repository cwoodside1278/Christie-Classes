#Christie Woodside
#Midterm for data Mining

'''Task 2: Logistic Regression'''
'''Last digit 9,0 Task 2:
• σx = 1
• ρ = 0.65
• β = [−1, 0.5, 0.5, 1, −0.7, 1]'''

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import statsmodels.api as sm
import matplotlib.pyplot as plt

'''Setting the parameters'''
N_sim = 5000 #5000 simulations
n = 750 #sample size n
std_dev = 1
rho_p = 0.65
beta = [-1, 0.5, 0.5, 1, -0.7, 1, 0] #added zero as the extra number for overparameterized model
mean_d = [0,0]
cov_matrix = [[std_dev**2, rho_p * std_dev**2], [rho_p * std_dev**2, std_dev**2]] #to get the covariance matric

#Counters to store the every time a models parameters is OUTSIDE the CI
total_outside_true_model = 0
total_outside_over_model = 0
total_param_true_model = 0
total_param_over_model = 0

#holds the estimated param values and whether they are within the 95% CI
params_true_model = []
params_over_model = []

# Initialize arrays to track counts for CI checks
true_in_CI_counts = np.zeros(len(beta) - 1)  # for true model (excluding constant)
over_in_CI_counts = np.zeros(len(beta))       # for overparameterized model (including constant)

inside_CI_true = np.zeros(len(beta) - 1)  # For true model
inside_CI_over = np.zeros(len(beta))      # For overparameterized model





'''Functions'''

def logistic_prob(mu):
    '''Function that calculates the logistical probability'''
    return 1 / (1 + np.exp(-mu))

def logistic_regression(X, Y):
    '''Function that will run logistical regression on the model 
    when its params are provided'''
    #Makes it less messy in sim_and_eval and more efficient
    X = sm.add_constant(X)  #adding an intercept
    model = sm.Logit(Y, X) #run logistical regression
    result = model.fit(disp=0)
    return result

def check_NOT_in_CI(true_params, ci, model_name):
    '''Check if parameters are NOT inside the confidence intervals'''
    count_outside = 0  # Initialize count outside as an integer

    for true_val, (lower, upper) in zip(true_params, ci):
        if not (lower <= true_val <= upper):
            count_outside += 1  # Increment the count if outside

    return count_outside  # Return the total count of parameters outside


def check_in_CI(true_params, ci, inside_CI_array):
    '''Check if true parameters are inside the confidence intervals'''
    for i, (true_val, (lower, upper)) in enumerate(zip(true_params, ci)):
        if lower <= true_val <= upper:
            inside_CI_array[i] += 1  # Increment the specific index for each parameter
    return inside_CI_array



def sim_and_eval_v2():
    '''This function is to simulate 5000 times. It creates the covarites and runs LOGISTICAL regression 
    on all three models. It also runs and stores metrics on all of the models to be used for analysis'''

    #Stating these variables as global so they will work
    global total_outside_true_model, total_outside_over_model, total_param_true_model, total_param_over_model
    global inside_CI_true, inside_CI_over

    for sim in range(N_sim):
        #create the covariates
        X1, X2 = np.random.multivariate_normal(mean_d, cov_matrix, n).T
        X3 = np.where(X1 * X2 < 0, 'A', 'B')  #new conditions for this covariate
        X3_a = (X3 == 'A').astype(int)

        #create the combined dataframe
        df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X3_a': X3_a #the indictaor for when X3=A
        })
        # Create interaction terms
        df['X3_a_X2'] = df['X3_a'] * df['X2']  # Interaction between X3_a and X2
        df['X3_a_X1'] = df['X3_a'] * df['X1']  # Interaction between X3_a and X1

        #Calculating Y for the True Model
        t_mu = (beta[0] + beta[1] * df['X1'] + beta[2] * df['X2'] + beta[3] * df['X3_a'] + 
          beta[4] * df['X1']**2 + beta[5] * df['X3_a_X2'])
        prob_t = logistic_prob(t_mu)
        t_Y = np.random.binomial(1, prob_t)   #the Y in binary for true model

        #Calculating the Y for overparameterized model
        o_mu = (beta[0] + beta[1] * df['X1'] + beta[2] * df['X2'] + beta[3] * df['X3_a'] + 
            beta[4] * df['X1']**2 + beta[5] * df['X3_a_X2'] + beta[6] * df['X3_a_X1'])
        prob_o = logistic_prob(o_mu) #calculatingt he logistical probability
        o_Y = np.random.binomial(1, prob_o)  #The Y in binary for the Overparamaterized model

        #Design matrix for the models for logistical regression
        true_design = np.column_stack([df['X1'], df['X2'], df['X3_a'], df['X1']**2, df['X3_a_X2']])
        over_design = np.column_stack([df['X1'], df['X2'], df['X3_a'], df['X1']**2, df['X3_a_X2'], df['X3_a_X1']])

        #Run logistical regression using the function for both of the models
        result_true = logistic_regression(true_design, t_Y)
        result_over = logistic_regression(over_design, o_Y)

        #Stroing the param estimate and the confidence intervals
        params_true = result_true.params
        params_over = result_over.params
        ci_true = result_true.conf_int() #for 95% CI
        ci_over = result_over.conf_int() #for 95% CI

        # Store parameter estimates in a list for analysis use
        params_true_model.append(result_true.params)
        params_over_model.append(result_over.params)

        # check/counter if the params are NOT within the ci
        total_outside_true_model += check_NOT_in_CI(params_true, ci_true, 'True Model')
        total_outside_over_model += check_NOT_in_CI(params_over, ci_over, 'Overparametrized Model')

        #original beta true params to check if they are NOT within the CI
        true_m_params = beta[:6]
        total_param_true_model += check_NOT_in_CI(true_m_params, ci_true, 'True Model')  
        total_param_over_model += check_NOT_in_CI(beta, ci_over, 'Overparametrized Model') 

        # Track counts for parameters that are WITHIN CIs
        inside_CI_true = check_in_CI(beta[:len(inside_CI_true)], ci_true, inside_CI_true)
        inside_CI_over = check_in_CI(beta, ci_over, inside_CI_over)
        # inside_CI_true += check_in_CI(true_m_params, ci_true, 'True Model')
        # inside_CI_over += check_in_CI(beta, ci_over, 'Overparameterized Model')

        # Track counts for CIs
        for i in range(len(params_true)):
            if ci_true[i][0] <= beta[i] <= ci_true[i][1]:
                true_in_CI_counts[i] += 1

        for i in range(len(params_over)):
            if ci_over[i][0] <= beta[i] <= ci_over[i][1]:
                over_in_CI_counts[i] += 1


        if sim < 5:  # print the first 5 simulation results to view
            print(f"Simulation {sim+1}:")
            print("True Model Parameter Estimates:")
            print(params_true)
            print("True Model Confidence Intervals:")
            print(ci_true)
            print("\nOverparameterized Model Parameter Estimates:")
            print(params_over)
            print("Overparameterized Model Confidence Intervals:")
            print(ci_over)
            print("-----------------------------------------------------------------------------------")



'''-------Summary Results--------'''
#Run the sim_and_eval_v2 function to get the results
sim_and_eval_v2()

#Prints outthe counter to see which model has the most parameters inside
print(f"\nTotal estimated parameters NOT INSIDE the confidence interval for True Model: {total_outside_true_model}")  #these were more as a check to verify that my code was working as intended
print(f"Total estimated parameters NOT INSIDE the confidence interval for Overparametrized Model: {total_outside_over_model}")
print(f"\nTotal estimated parameters INSIDE the confidence interval for True Model: {N_sim - total_outside_true_model}")
print(f"Total estimated parameters INSIDE the confidence interval for Overparametrized Model: {N_sim - total_outside_over_model}")

print(f"\nTotal -true beta- parameters NOT INSIDE the confidence interval for True Model: {total_param_true_model}") #These numbers are more important and useful for analysis
print(f"Total -true beta- parameters NOT INSIDE the confidence interval for Overparametrized Model: {total_param_over_model}")
print(f"\nTotal -true beta- parameters that are INSIDE the confidence interval for True Model: {N_sim - total_param_true_model}")
print(f"Total -true beta- parameters that are INSIDE the confidence interval for Overparametrized Model: {N_sim - total_param_over_model}")

print("\n-------------------------------------------------------------")


#convert to DataFrames 
params_true_df = pd.DataFrame(params_true_model, columns=['Intercept', 'X1', 'X2', 'X3_a', 'X1_squared', 'X3_a_X2'])
params_over_df = pd.DataFrame(params_over_model, columns=['Intercept', 'X1', 'X2', 'X3_a', 'X1_squared', 'X3_a_X2', 'X3_a_X1'])

# calculating Mean Squared Errors (MSE) for Parameters
true_params = np.array(beta[:6])  # true parameters for the true model so one less index
over_params = np.array(beta[:7]) #all indexes for overparameterized

mse_true_model = ((params_true_df - true_params) ** 2).mean() #mean MSE value
mse_over_model = ((params_over_df - over_params) ** 2).mean()

print("\nMean Squared Errors for Parameters:\n")
print("True Model MSE:")
print(mse_true_model)
print("\nOverparameterized Model MSE:")
print(mse_over_model)
print("\n-------------------------------------------------------------")


# print("Inside CI counts for True Model:", inside_CI_true)
# print("Inside CI counts for Overparameterized Model:", inside_CI_over)
# Calculate the relative frequency of true values inside the confidence intervals
relative_freq_true = [(inside / N_sim) for inside in inside_CI_true]
relative_freq_over = [(inside / N_sim) for inside in inside_CI_over]

print("\nRelative Frequency of True Parameters Inside CI:")
print("True Model:", relative_freq_true)  
print("Overparameterized Model:", relative_freq_over)
print("\n-------------------------------------------------------------")


'''---------------------------------Plotting----------------------------------------------------'''

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(11, 6)) 
axes = axes.flatten()  # flattening the array for easier indexing.

#histograms for each column in params_true_df for TRUE MODEL using estimated paras 
for i, column in enumerate(params_true_df.columns):
    params_true_df[column].hist(bins=30, alpha=0.7, color='red', ax=axes[i])
    axes[i].set_title(f'Histogram of {column}')  
    axes[i].set_xlabel('Parameter Estimates')    
    axes[i].set_ylabel('Frequency')              


plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(11, 6))  
axes = axes.flatten() 

#histograms for each column in params_over_df for OVER MODEL using estimated params
for i, column in enumerate(params_over_df.columns):
    params_over_df[column].hist(bins=30, alpha=0.7, color='blue', ax=axes[i])
    axes[i].set_title(f'Histogram of {column}')  
    axes[i].set_xlabel('Parameter Estimates')    
    axes[i].set_ylabel('Frequency')              

plt.tight_layout()
plt.show()




