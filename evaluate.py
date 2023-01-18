import matplotlib.pyplot as plt
from math import sqrt

def plot_residuals(y, y_hat):
    '''
    This function takes in predicitions and y values then uses a scatterplot to graph the residuals for the data entered
    '''
    plt.scatter(y, (y_hat-y))
    plt.xlabel('y_hat')
    plt.ylabel('Residuals')
    plt.show()

def regression_errors(y, y_hat):
    SSE = sum((y_hat - y) ** 2)
    ESS = sum((y_hat - y.mean()) ** 2)
    TSS = ESS + SSE
    MSE = SSE/len(y_hat)
    RMSE = sqrt(MSE)
    return print(f'{SSE} is the SSE \n {ESS} is the ESS \n {TSS} is the TSS \n {MSE} is the MSE \n {RMSE} is the RMSE')

def baseline_mean_errors(y):
    SSE = sum((y- y.mean()) ** 2)
    MSE = SSE/len(y)
    RMSE = sqrt(MSE)
    return print(f'{SSE} is the SSE \n {MSE} is the MSE \n {RMSE} is the RMSE')

def better_than_baseline(y, y_hat):
    test = sum((y - y.mean()) ** 2) - sum((y_hat - y) ** 2)
    if test > 0:
        print('The model is better than the baseline')
    else:
        print('The model is not better than the baseline')