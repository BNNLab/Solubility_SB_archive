'''
This python script calculates the metrics when test set melting point is random deviated upwards or downwards by certain amounts.
An ExtraTrees model is created for each deviation amount and the metrics saved.
See AddingMPError2.py for the effect on predictions when all test set MPs are deviated up, down or not at all.
INPUTS:
Dataset.csv - a .csv file with the descriptors in
OUTPUTS:
MPdev_metrics1 - a .csv file detailing the metrics at different magnitudes of MP deviation
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

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of descriptor file
Dataset_name="Dataset.csv"
output_metrics=os.path.join(dir,"MPdev_metrics1.csv")#location of output file for ML metrics for MP deviations

##section 3: define methods for getting ML results
##list of deviations to apply to test set MP (randomly upwards or downwards)
dev_list=[0,5,10,15,20,25,30,35,40,50,60,70,80]
#define RMSE
def rmse(predictions, targets):
	return np.sqrt(((predictions - targets) ** 2).mean())
#define method to find predictions within certain range
def within_range(list1, list2, range2):
	x=0
	for i in range(len(list2)):
		if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
			x+=1
	return((float(x)/(len(list2)))*100)
#Stat Split Method
def stat_split_metrics(train,test,mp_dev):
	test2=test
	RMSE=[]
	R2=[]
	N1=[]
	N05=[]
	#place target value in y
	y_train = train['LogS']
	y_test = test['LogS']
	y_test=np.array(y_test)
	#######deviate m.p. by value given and reintroduce for test set
	MP=test2["MP"]
	MP=np.array(MP)
	rand=np.random.randint(2, size=len(MP))
	for i in range(len(MP)):
		if rand[i]==0:
			MP[i]=MP[i]-mp_dev
		if rand[i]==1:
			MP[i]=MP[i]+mp_dev
	test2["MP_dev"]=MP
	#place descriptors in X
	X_train = train[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
					 'LsoluHsolv','LsolvHsolu','SASA','O_charges',
					 'C_charges','Most_neg','Most_pos','Het_charges']]
	X_test = test2[['MW','MP_dev','Volume','G_solv','DeltaG_sol','solv_dip',
					 'LsoluHsolv','LsolvHsolu','SASA','O_charges',
					 'C_charges','Most_neg','Most_pos','Het_charges']]
	#scale data
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	#run models
	#ExtraTrees
	tree3 = ensemble.ExtraTreesRegressor(n_estimators=500)
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
	#create dataframe of metrics for deviation and send back
	Metrics=["ExtraTrees",mp_dev,R2_2[0],RMSE[0],N1[0],N05[0]]
	return(Metrics)
##putting it all together
def getMPdevs(Dataset_name,output_metrics):
	##list of metrics
	mets_list=[]
	##for every deviation
	for devs in dev_list:
		##reload dataset
		Dataset=pd.read_csv(os.path.join(dir,Dataset_name))
		##get train and test sets
		train=Dataset[Dataset["Train_test"]=="Train"]
		test=Dataset[Dataset["Train_test"]=="Test"]
		##append the metrics for the deviation to the test set
		mets_list.append(stat_split_metrics(train,test,devs))
	##make dataframe and save
	mets_list=pd.DataFrame(data=mets_list,columns=["Model","MP Deviation","R2","RMSE","% Within 1","% Within 0.7"])
	mets_list.to_csv(output_metrics,index=False)

##section 4: run method and save file
getMPdevs(Dataset_name,output_metrics)
