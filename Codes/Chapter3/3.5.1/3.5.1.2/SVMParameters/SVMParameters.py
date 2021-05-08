'''
This python script find the best parameters for SVM for a dataset.
The method involves optimising epsilon, gamma and C in turn using 10-fold cross validation of the training set.
INPUTS:
Dataset.csv - a .csv file with the data in. The columns must be named in the same way as below
OUTPUTS:
Dataset_params.csv - a .csv file containing the optimised parameters for epsilon, gamma and C
'''
##section 1: import modules
from sklearn.model_selection import GridSearchCV
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import svm

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of descriptor file
output_params=os.path.join(dir,"Dataset_params.csv")#location of output file for SVM parameters

##section 3: define methods
##just use training data to get SVM parameters
train=Dataset[Dataset["Train_test"]=="Train"]
##descriptors to use in the optimisation
descs=['MW','MP','Volume','E0_gas','E0_solv','DeltaE0_sol',
                     'G_gas','G_solv','DeltaG_sol','gas_dip','solv_dip',
                     'HOMO','LUMO','LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges','N_atoms']
def grid_search(train,param_grid):
    #place target value in y
    y_train = train['LogS']
    #place descriptors in X
    X_train = train[descs]
    #scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    #grid search using 10-fold CV, return best parameters
    svm2 = svm.SVR()
    gs=GridSearchCV(estimator=svm2,param_grid=param_grid,cv=10)
    gs=gs.fit(X_train,y_train)
    return(gs.best_params_)
##a search of SVM parameter space placed the parameters in these regions
epsilon_params=[0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
gamma_params=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1]
C_params=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
def getBestParams(train,output_params):
	##find best epsilon
	param_grid_epsilon={'epsilon':epsilon_params}
	best_epsilon_param=grid_search(train,param_grid_epsilon)
	epsilon=best_epsilon_param['epsilon']
	##find best gamma
	param_grid_gamma={'epsilon':[epsilon],'gamma':gamma_params}
	best_gamma_param=grid_search(train,param_grid_gamma)
	gamma=best_gamma_param['gamma']
	##find best C
	param_grid_C={'epsilon':[epsilon],'gamma':[gamma],'C':C_params}
	best_C_param=grid_search(train,param_grid_C)
	C=best_C_param['C']
	##make into dataframe and save
	params_df=pd.DataFrame(data=[[epsilon,gamma,C]],columns=["epsilon","gamma","C"])
	params_df.to_csv(output_params,index=False)

##section 4: run methods to get parameters and save
getBestParams(train,output_params)