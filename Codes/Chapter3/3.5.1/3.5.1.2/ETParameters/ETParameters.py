'''
This python script returns metrics for different combinations of number of trees used in ExtraTrees models.
A model is built for every combination, trained on the train data and then tested on the test data.
the average metrics (and the SD) of 100 runs is outputted to file.
INPUTS:
Dataset.csv - a .csv file with the data in. The columns must be named in the same way as below
OUTPUTS:
Dataset_ntrees.csv - a .csv file containing the average metrics and SD for every tree combination
'''
##section 1: import modules
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import ensemble
from scipy.stats import pearsonr
import math
import statistics

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of descriptor file
output_params=os.path.join(dir,"Dataset_ntrees.csv")#location of output file for metrics for different numbers of trees

##section 3: methods for getting metrics for numbers of trees
##split the data into train and test (train on train data and record metric for different number of trees using test data)
train=Dataset[Dataset["Train_test"]=="Train"]
test=Dataset[Dataset["Train_test"]=="Test"]
#define numbers of trees to test
n_trees1=np.arange(1,10,1)########
n_trees2=np.arange(10,110,10)
n_trees3=np.arange(200,1100,100)
n_trees4=np.arange(2000,6000,1000)
n_trees=[]
n_trees.extend(n_trees1)
n_trees.extend(n_trees2)
n_trees.extend(n_trees3)
n_trees.extend(n_trees4)
n_trees=np.array(n_trees)
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
##define method to run ET model
def stat_split_metrics(train,test,n_trees):
	##initiate lists to place metrics
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
	#ExtraTrees
	##set up with required number of trees
	tree3 = ensemble.ExtraTreesRegressor(n_estimators=n_trees,n_jobs=-1)
	##fit and predict
	tree3.fit(X_train, y_train)
	tree3preds = tree3.predict(X_test)
	y_test=np.array(y_test)
	#evaluate model and add metrics to lists
	R2.append(pearsonr(tree3preds, y_test))
	RMSE.append(rmse(tree3preds, y_test))
	N1.append(within_range(y_test,tree3preds,1))
	N05.append(within_range(y_test,tree3preds,0.7))
	#get R2 from Pearson output
	R2_2=[]
	for i in range(len(R2)):
		x=re.findall('\d\.\d+',str(R2[i]))
		j=float(x[0])
		j=j**2
		R2_2.append(j)
	#create dataframe of metrics
	Metrics=[]
	Metrics.append("ET")
	Metrics.append(n_trees)
	Metrics.append(R2_2[0])
	Metrics.append(RMSE[0])
	Metrics.append(N1[0])
	Metrics.append(N05[0])
	#return these metrics
	return(Metrics)
##method to get average metrics for each number of trees with the SD as error
def get_cons(metrics,n_trees):
    #Get mean metrics and the SD associated with them
    mean_metrics2 = metrics[["R2","RMSE","% within 1.0","% within 0.7"]]
    mean_metrics = list(mean_metrics2.mean())
    std_metrics = list(mean_metrics2.std())
    mean_metrics = mean_metrics + std_metrics
    mean_metrics.insert(0,n_trees)
    mean_metrics.insert(0,"ET")
    return(mean_metrics)
##define method to put it all together
def getTreeMetrics(n_trees,n_rep,train,test,Output):
	##master list
	master=[]
	##for every tree combination
	for f in n_trees:
		##new list
		metrics_all=[]
		##for so many repetitions 
		for g in range(n_rep):
			##get metrics
			metrics=stat_split_metrics(train,test,f)
			##appned to list
			metrics_all.append(metrics)
		#columns names for intermediate pandas dataframe
		#make dataframe
		metrics_all=pd.DataFrame(data=metrics_all,columns=["Method","n_trees","R2","RMSE","% within 1.0","% within 0.7"])
		#get metrics of mean predictions
		metrics=get_cons(metrics_all,f)
		##add these metrics to master dataframe
		master.append(metrics)
	##make new dataframe from final results
	metrics_df=pd.DataFrame(data=master,columns=["Method","n_trees","R2","RMSE","% within 1.0","% within 0.7","R2_std",
														"RMSE_std","% 1.0 std","% 0.7 std"])
	metrics_df.to_csv(Output,index=False)

##section 4: run method to get average metrics across 100 runs for each tree combination and save
getTreeMetrics(n_trees,n_rep,train,test,output_params)
