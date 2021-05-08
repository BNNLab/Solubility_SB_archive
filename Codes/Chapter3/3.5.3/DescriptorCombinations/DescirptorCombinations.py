'''
This python script run ExtraTrees models for every combination of 11, 12, 13 and 14 descriptors (leaving out 1-3 descriptors).
Each model is trained on the training data and tested on the test data. The metrics and descriptors used for that model are recorded.
INPUTS:
Dataset.csv - a .csv file with the descriptors in, named as below
OUTPUTS:
Dataset_combos - .csv file detailing the descriptors removed and metrics for every combination of descriptors explored
'''
##section 1: import modules
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn import ensemble
from scipy.stats import pearsonr
import math
import statistics
from itertools import combinations

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of Dataset file
output_combos=os.path.join(dir,"Dataset_combos.csv")#location of output file for descriptor combinations

##section 3: define methods for getting ML results
##split into train and test (train on the former, test on the latter)
train=Dataset[Dataset["Train_test"]=="Train"]
test=Dataset[Dataset["Train_test"]=="Test"]
##names of descriptor columns
descs=['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip','LsoluHsolv','LsolvHsolu','SASA','O_charges','C_charges','Most_neg',
 'Most_pos','Het_charges']
##list of number of descriptors to consider 11, 12, 13 and 14 (missing out 1, 2 or 3 descriptors)
num_list=np.arange(11,15,1)
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
#define method to get metrics from the descriptor combination
def stat_split_metrics(train,test,descs2):
	RMSE=[]
	R2=[]
	N1=[]
	N05=[]
	#place target value in y
	y_train = train['LogS']
	y_test = test['LogS']
	y_test=np.array(y_test)
	#place descriptors in X
	X_train = train[descs2]
	X_test = test[descs2]
	#scale data
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	#run models
	#ExtraTrees
	tree3 = ensemble.ExtraTreesRegressor(n_estimators=500,n_jobs=-1)
	tree3.fit(X_train, y_train)
	tree3preds = tree3.predict(X_test)
	#evaluate model
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
	removed=['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip','LsoluHsolv','LsolvHsolu','SASA','O_charges','C_charges','Most_neg',
			 'Most_pos','Het_charges']
	for i in descs2:
		removed.remove(i)
	#add metrics to list and return list
	Metrics=[]
	Metrics.append(descs2)
	Metrics.append(removed)
	Metrics.append("ET")
	Metrics.append(R2_2[0])
	Metrics.append(RMSE[0])
	Metrics.append(N1[0])
	Metrics.append(N05[0])
	return(Metrics)
##method for putting it all together
def getCombos(train,test,output_combos):
	##master list
	master=[]
	##for every number of descriptors
	for f in num_list:
		##get every unique combination of descriptors
		perm = combinations(descs, int(f))
		##get the metrics for these combinations
		for g in list(perm):
			mets=stat_split_metrics(train,test,list(g))
			master.append(mets)
	##make into dataframe and save
	metrics=pd.DataFrame(data=master,columns=['Descs','Removed','Model','R2','RMSE','% within 1','% within 0.5'])
	metrics.to_csv(output_combos,index=False)

##section 4: run method to get metrics for every combination
getCombos(train,test,output_combos)