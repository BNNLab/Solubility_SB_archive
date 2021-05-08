'''
This python script returns the mean importance for every descriptor across 100 runs (and SD) for ExtraTrees model.
INPUTS:
Dataset.csv - a .csv file with the descriptors in, named as below
OUTPUTS:
Dataset_combos - .csv file with the mean importance of every descriptor and and SD (100 runs)
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
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of Dataset file
output_importance=os.path.join(dir,"Dataset_importance.csv")#location of output file for descriptor importance

##section 3: define methods for getting ML results
##split into train and test (train on the former, test on the latter)
train=Dataset[Dataset["Train_test"]=="Train"]
test=Dataset[Dataset["Train_test"]=="Test"]
descs=['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
					 'LsoluHsolv','LsolvHsolu','SASA','O_charges',
					 'C_charges','Most_neg','Most_pos','Het_charges']
#number of repeats for SD errors
n_rep=100 ###########
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
#Return feature importance for a run
def stat_split_metrics(train,test):
	#place target value in y
	y_train = train['LogS']
	y_test = test['LogS']
	y_test=np.array(y_test)
	#place descriptors in X
	X_train = train[descs]
	X_test = test[descs]
	#scale data
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	#run models
	#ExtraTrees
	tree3 = ensemble.ExtraTreesRegressor(n_estimators=500,n_jobs=-1)
	tree3.fit(X_train, y_train)
	tree3preds = tree3.predict(X_test)
	#return feature importances
	return(tree3.feature_importances_)
#get mean metrics from these predictions
def get_cons(metrics):
	#Get mean metrics
	lst=[]
	for f in descs:
		temp=[]
		metrics2=metrics[f]
		metrics2=np.array(metrics2)
		mean=np.mean(metrics2)
		std=np.std(metrics2)
		temp.append(f)
		temp.append(mean)
		temp.append(std)
		lst.append(temp)
	mean_metrics=pd.DataFrame(data=lst,columns=["Descriptor","Mean Importance","Std"])
	return(mean_metrics)
#putting it all together
def getImp(train,test,output_importance):
	##master list
	master=[]
	for g in range(n_rep):
		##for each repeat
		features=stat_split_metrics(train, test)
		##append importance
		master.append(features
	##make into dataframe
	df=pd.DataFrame(data=master,columns=descs)
	##convert to mean importance with SD
	df=get_cons(df)
	##save to file
	df.to_csv(output_importance,index=False)

##section 4: run method and save importance
getImp(train,test,output_importance)
