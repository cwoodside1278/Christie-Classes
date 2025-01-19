#Christie Woodside
#Midterm for Data Mining class

'''Midterm Task 1 Multiple Linear Regression'''
'''Last digit 9,0 Task 1: 
• σx = 0.85 
• ρ = 0.85 
• β = [1, −1.4, 0.6, 0.2, −0.2, 1] 
• σe = 1 
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

pd.set_option('display.max_rows', None)     # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.width', None)        # Display full width in terminal
pd.set_option('display.max_colwidth', None) # Set max column width


'''Setting up the parameters'''
N_sim = 5000 #5000 simulations
n = 750 #sample size n
mean_d = [0,0]
std_dev = 0.85
rho_p = 0.85
sigma_e = 1
beta = [1, -1.4, 0.6, 0.2, -0.2, 1] #co efficients for the true model
cov_matrix = [[std_dev**2, rho_p * std_dev**2], [rho_p * std_dev**2, std_dev**2]] #to get the covariance matric


#Functions for use in analysis
def include_interactions(df):
    '''Creates the interaction terms for the enlarged model for eahc simulation'''
    df['X2_squared'] = df['X2'] ** 2
    df['X1_X3A'] = df['X1'] * df['X3_A']
    df['X1_squared'] = df['X1'] ** 2 
    df['X2_X3A'] = df['X2'] * df['X3_A'] 
    return df

def kfold_X_Val(X, y, k=10):
    """Function that will perform K-fold cross-validation. It returns MSE."""
    '''Function created for Variable Selection Task Section of Task 1'''
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_val = []
    model = LinearRegression()

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mse_val.append(mse)
        
    return np.mean(mse_val)

#Used to store the values for results and analysis of the performance of the models
performance_simulations_metrics = {
    'true_model_r2': [],
    'reduced_model_r2': [],
    'enlarged_model_r2': [],
    'true_model_aic': [],
    'reduced_model_aic': [],
    'enlarged_model_aic': [],
    'true_model_mse': [],
    'reduced_model_mse': [],
    'enlarged_model_mse': []
}

#To store the residual and fitted values for the models after linear regression
#This was itis easier to plot in the analysis below
model_values = {
    'true_model_fitted': [],
    'true_model_residuals': [],
    'reduced_model_fitted': [],
    'reduced_model_residuals': [],
    'enlarged_model_fitted': [],
    'enlarged_model_residuals': []
}

# Used to store the total counts of model selection based on lowest AIC and lowest MSE
model_selection_counts = {
    'aic_true_model': 0,
    'aic_reduced_model': 0,
    'aic_enlarged_model': 0,
    'mse_true_model': 0,
    'mse_reduced_model': 0,
    'mse_enlarged_model': 0
}


def sim_and_eval():
    '''This function is to simulate 5000 times. It creates the covarites and runs linear regression 
    on all three models. It also runs and stores metrics on all of the models to be used for analysis'''
    for sim in range(N_sim):
        #create the covariates
        X1, X2 = np.random.multivariate_normal(mean_d, cov_matrix, n).T
        X3 = np.where(X1 < 0, 'A', 'B')
        X3_a = (X3 == 'A').astype(int)

        #create the combined dataframe
        df = pd.DataFrame({
            'X1': X1,
            'X2': X2,
            'X3_A': X3_a #the indictaor for when X3=A
        })

        #generate the mu (u) for the True Model
        True_model_mu = (beta[0] + beta[1] * df['X1'] + beta[2] * df['X2'] + 
                beta[3] * df['X3_A'] + beta[4] * df['X2'] ** 2 + 
                beta[5] * df['X3_A'] * df['X1'])
        
        epsilon = np.random.normal(0, sigma_e, n) #setting the epsilon number
        Y = True_model_mu + epsilon #create the response variable Y
        df['Y'] = Y  # add response variable Y to df

        #For englarged model, add the interaction terms from the method
        df = include_interactions(df)

        # Thhe three Models:
        # True Model: µ = β0 + β1X1 + β2X2 + β3 · I(X3 = A) + β4X2**2 + β5 · I(X3 = A) · X1
        true_model = sm.OLS(df['Y'], sm.add_constant(df[['X1', 'X2', 'X3_A', 'X2_squared', 'X1_X3A']])).fit()

        # Reduced Model: µ = β0 + β1X1 + β2X2 + β3 · I(X3 = A)
        reduced_model = sm.OLS(df['Y'], sm.add_constant(df[['X1', 'X2', 'X3_A']])).fit()
        
        # Enlarged Model: µ = β0+β1X1+β2X2+β3·I(X3 = A)+β4X2**2+β5·I(X3 = A)·X1+β6X1**2+β7·I(X3 = A)·X2
        enlarged_model = sm.OLS(df['Y'], sm.add_constant(df[['X1', 'X2', 'X3_A', 'X2_squared', 'X1_X3A', 'X1_squared', 'X2_X3A']])).fit()



        # Save fitted values and residuals for each model to plot later
        model_values['true_model_fitted'] = true_model.fittedvalues
        model_values['true_model_residuals'] = true_model.resid
        model_values['reduced_model_fitted'] = reduced_model.fittedvalues
        model_values['reduced_model_residuals'] = reduced_model.resid
        model_values['enlarged_model_fitted'] = enlarged_model.fittedvalues
        model_values['enlarged_model_residuals'] = enlarged_model.resid
        
        #Storing this simulations R2 value for eahc model
        performance_simulations_metrics['true_model_r2'].append(true_model.rsquared)
        performance_simulations_metrics['reduced_model_r2'].append(reduced_model.rsquared)
        performance_simulations_metrics['enlarged_model_r2'].append(enlarged_model.rsquared)
        #storing this simulations AIC value for each model
        performance_simulations_metrics['true_model_aic'].append(true_model.aic)
        performance_simulations_metrics['reduced_model_aic'].append(reduced_model.aic)
        performance_simulations_metrics['enlarged_model_aic'].append(enlarged_model.aic)

        # cross-validation for MSE for Variable Selection task
        X_true = df[['X1', 'X2', 'X3_A', 'X2_squared', 'X1_X3A']].values
        X_reduced = df[['X1', 'X2', 'X3_A']].values
        X_enlarged = df[['X1', 'X2', 'X3_A', 'X2_squared', 'X1_X3A', 'X1_squared', 'X2_X3A']].values
        y = df['Y'].values

        performance_simulations_metrics['true_model_mse'].append(kfold_X_Val(X_true, y))
        performance_simulations_metrics['reduced_model_mse'].append(kfold_X_Val(X_reduced, y))
        performance_simulations_metrics['enlarged_model_mse'].append(kfold_X_Val(X_enlarged, y))


        # best model selection based on lowest AIC, adding to the counter 'model_selection_counts'
        aic_values = [true_model.aic, reduced_model.aic, enlarged_model.aic]
        min_aic_index = np.argmin(aic_values)

        if min_aic_index == 0:
            model_selection_counts['aic_true_model'] += 1
        elif min_aic_index == 1:
            model_selection_counts['aic_reduced_model'] += 1
        else:
            model_selection_counts['aic_enlarged_model'] += 1

        # best model selection based on lowest cross-validated MSE, adding to the counter 'model_selection_counts'
        mse_values = [
            performance_simulations_metrics['true_model_mse'][-1], 
            performance_simulations_metrics['reduced_model_mse'][-1], 
            performance_simulations_metrics['enlarged_model_mse'][-1]
        ]
        min_mse_index = np.argmin(mse_values)

        if min_mse_index == 0:
            model_selection_counts['mse_true_model'] += 1
        elif min_mse_index == 1:
            model_selection_counts['mse_reduced_model'] += 1
        else:
            model_selection_counts['mse_enlarged_model'] += 1




'''Results from the 'sim_and_eval' function'''
#run the function
sim_and_eval()

# Convert the list of metrics to DataFrame for analysis
metrics_df = pd.DataFrame(performance_simulations_metrics)
print(metrics_df.describe())

#Printing the model selection counts to see which is best
print("\n\nModel Selection Counts based on lowest AIC value:")
print(f"True Model: {model_selection_counts['aic_true_model']}")
print(f"Reduced Model: {model_selection_counts['aic_reduced_model']}")
print(f"Enlarged Model: {model_selection_counts['aic_enlarged_model']}")

print("\nModel Selection Counts based on the lowest cross-validated MSE:")
print(f"True Model: {model_selection_counts['mse_true_model']}")
print(f"Reduced Model: {model_selection_counts['mse_reduced_model']}")
print(f"Enlarged Model: {model_selection_counts['mse_enlarged_model']}")

#Display the time taken and parameter estimates
# print(f"Time taken to run {N_sim} simulations: {elapsed_time:.2f} seconds \n \n ")


'''-----------------Plots for Analysis-----------------'''
'''Plotting the R2 statistic information and Residuals to compare the performance and metrics of the models '''

###### Plotting the R2 statistic to evaluate the metric across all the simulations for eahc model
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# plot the histogram for each model's R-squared values
sns.histplot(metrics_df['true_model_r2'], color="blue", label="True Model R²", kde=True, bins=30)
sns.histplot(metrics_df['reduced_model_r2'], color="green", label="Reduced Model R²", kde=True, bins=30)
sns.histplot(metrics_df['enlarged_model_r2'], color="red", label="Enlarged Model R²", kde=True, bins=30)

plt.title("Distribution of $R^2$ Values Across Simulations", fontsize=16)
plt.xlabel("$R^2$ Value", fontsize=14)
plt.ylabel("Frequency", fontsize=14)

plt.legend()
plt.show()


######## Residual vs fitted values plot
plt.figure(figsize=(15, 10))
# Residual vs Fitted for True Model
plt.subplot(3, 1, 1)
sns.scatterplot(x= model_values['true_model_fitted'], y= model_values['true_model_residuals'], color="blue", label="True Model")
plt.axhline(0, color='black', linestyle='--', linewidth=1)  # Line at 0 for reference
plt.title("Residuals vs Fitted Values: True Model", fontsize=16)
plt.xlabel("Fitted Values", fontsize=14)
plt.ylabel("Residuals", fontsize=14)

# Residual vs Fitted for Reduced Model
plt.subplot(3, 1, 2)
sns.scatterplot(x=model_values['reduced_model_fitted'], y=model_values['reduced_model_residuals'], color="green", label="Reduced Model")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Residuals vs Fitted Values: Reduced Model", fontsize=16)
plt.xlabel("Fitted Values", fontsize=14)
plt.ylabel("Residuals", fontsize=14)

# Residual vs Fitted for Enlarged Model
plt.subplot(3, 1, 3)
sns.scatterplot(x=model_values['enlarged_model_fitted'], y=model_values['enlarged_model_residuals'], color="red", label="Enlarged Model")
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.title("Residuals vs Fitted Values: Enlarged Model", fontsize=16)
plt.xlabel("Fitted Values", fontsize=14)
plt.ylabel("Residuals", fontsize=14)

plt.tight_layout()
plt.show()


##### Plotting the residual distributions for True, Reduced, and Enlarged model
plt.figure(figsize=(10, 6))
# Residual distribution for True Model
sns.kdeplot(model_values['true_model_residuals'], color="blue", label="True Model", fill=True)
# Residual distribution for Reduced Model
sns.kdeplot(model_values['reduced_model_residuals'], color="green", label="Reduced Model", fill=True)
# Residual distribution for Enlarged Model
sns.kdeplot(model_values['enlarged_model_residuals'], color="red", label="Enlarged Model", fill=True)

plt.title("Residual Distribution Plot for True, Reduced, and Enlarged Models", fontsize=16)
plt.xlabel("Residuals", fontsize=14)
plt.ylabel("Density", fontsize=14)

plt.legend()
plt.show()
