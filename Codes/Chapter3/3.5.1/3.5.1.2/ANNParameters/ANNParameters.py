'''
This python script returns metrics for different number of nodes in a single hidden layer for ANN.
The number of nodes metrics are averaged for up to 100 runs (if network fails to run it tries again, only keeping combinations that give 3 or more converged networks).
INPUTS:
Dataset.csv - a .csv file with the data in. The columns must be named in the same way as below
OUTPUTS:
Dataset_nlayers.csv - a .csv file containing the average metrics and SD for every number of nodes combination
'''
##section 1: import modules
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from scipy.stats import pearsonr
import math
import statistics
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter('error', category=ConvergenceWarning)#filter convergence warnings
warnings.simplefilter('ignore', category=DataConversionWarning)

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of descriptor file
output_params=os.path.join(dir,"Dataset_nlayers.csv")#location of output file for metrics for different numbers of layers

##section 3: methods for getting metrics for numbers of layers
##split the data into train and test (train on train data and record metric for different number of layers using test data)
train=Dataset[Dataset["Train_test"]=="Train"]
test=Dataset[Dataset["Train_test"]=="Test"]
#define numbers of layer sizes to test
#number of layers
n_layers1=np.arange(1,10,1)########
n_layers2=np.arange(10,110,10)
n_layers3=np.arange(200,1100,100)
n_layers4=np.arange(2000,6000,1000)
n_layers=[]
n_layers.extend(n_layers1)
n_layers.extend(n_layers2)
n_layers.extend(n_layers3)
n_layers.extend(n_layers4)
n_layers=np.array(n_layers)
#number of repeats for errors (given as 1 SD of the metrics)
n_rep=100###########
#define metrics
#define RMSE
def rmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean())
#define % within certain range
def within_range(list1, list2, range2):
	x=0
	for i in range(len(list2)):
		if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
			x+=1
	return((float(x)/(len(list2)))*100)
#define getting R2 method
def get_R2(R2):
	R2_2=[]
	for i in range(len(R2)):
		x=re.findall('\d\.\d+',str(R2[i]))
		j=float(x[0])
		j=j**2
		R2_2.append(j)
	return(R2_2)
##descriptors used
descs=['MW','MP','Volume','E0_gas','E0_solv','DeltaE0_sol',
					 'G_gas','G_solv','DeltaG_sol','gas_dip','solv_dip',
					 'HOMO','LUMO','LsoluHsolv','LsolvHsolu','SASA','O_charges',
					 'C_charges','Most_neg','Most_pos','Het_charges','N_atoms']
#method to run MLP
#Stat Split Method
def stat_split_metrics(train,test,n_layers):
	RMSE=[]
	R2=[]
	N1=[]
	N05=[]
	#place target value in y
	y_train = train['LogS']
	y_test = test['LogS']
	#place descriptors in X
	X_train = train[descs]
	X_test = test[descs]
	#scale data
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	#run models
	#ANN
	mlp = MLPRegressor(hidden_layer_sizes=n_layers,max_iter=1000)
	mlp.fit(X_train, y_train)   
	mlp2preds = mlp.predict(X_test)
	y_test = np.array(y_test)
	if np.ptp(mlp2preds) == 0:
		return()
	#evaluate model
	R2.append(pearsonr(mlp2preds, y_test))
	RMSE.append(rmse(mlp2preds, y_test))
	N1.append(within_range(y_test,mlp2preds,1))
	N05.append(within_range(y_test,mlp2preds,0.7))
	#get R2 from Pearson output
	R2_2=[]
	for i in range(len(R2)):
		x=re.findall('\d\.\d+',str(R2[i]))
		j=float(x[0])
		j=j**2
		R2_2.append(j)
	#create dataframe of metrics
	Metrics=[]
	Metrics.append("ANN")
	Metrics.append(n_layers)
	Metrics.append(R2_2[0])
	Metrics.append(RMSE[0])
	Metrics.append(N1[0])
	Metrics.append(N05[0])
	return(Metrics)
#get mean metrics from these predictions
#needs to be in pandas dataframe
def get_cons(metrics,n_layers):
	#Get mean metrics
	mean_metrics2 = metrics[["R2","RMSE","% within 1.0","% within 0.7"]]
	mean_metrics = list(mean_metrics2.mean())
	std_metrics = list(mean_metrics2.std())
	mean_metrics = mean_metrics + std_metrics
	mean_metrics.insert(0,n_layers)
	mean_metrics.insert(0,"ANN")
	return(mean_metrics)
def getLayers(n_layers,n_rep,train,test,Output):
	master=[]
	for f in n_layers:
		metrics_all=[]
		for g in range(n_rep):
			try:
				metrics=stat_split_metrics(train,test,f)
			except:
				break
			metrics_all.append(metrics)
		#columns names for intermediate pandas dataframe
		#make dataframe
		metrics_all=pd.DataFrame(data=metrics_all,columns=["Method","n_layers","R2","RMSE","% within 1.0","% within 0.7"])
		if len(metrics_all["R2"])<3:
			continue
		#get metrics of mean predictions
		metrics=get_cons(metrics_all,f)
		master.append(metrics)
	metrics_df=pd.DataFrame(data=master,columns=["Method","n_layers","R2","RMSE","% within 1.0","% within 0.7","R2_std",
														"RMSE_std","% 1.0 std","% 0.7 std"])
	metrics_df.to_csv(Output,index=False)

##section 4: run method and get number of layers
getLayers(n_layers,n_rep,train,test,output_params)