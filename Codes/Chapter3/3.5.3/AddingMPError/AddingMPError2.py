'''
This python script records the predictions and metrics for deviation test set MP all upwards or all downwards.
An ExtraTrees model is created for each deviation direction and amount and the metrics/predictions saved.
See AddingMPError1.py for the effect on predictions when test set molecules are deviated at random upwards or downwards for each deviation.
INPUTS:
Dataset.csv - a .csv file with the descriptors in
OUTPUTS:
MPdev_metrics2 - a .csv file detailing the metrics at different magnitudes of MP deviation
MPdev_preds2 - a .csv file detailing the predictions at different magnitudes of MP deviation
'''
#section 1: import modules
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
output_metrics=os.path.join(dir,"MPdev_metrics2.csv")#location of output file for ML metrics for MP deviations
output_preds=os.path.join(dir,"MPdev_preds2.csv")#location of output file for ML predictions for MP deviations

##section 3: methods to run deviation and return predictions and metrics
##list of deviations to apply to test set MP (both all upwards or all downwards)
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
def stat_split_metrics(train,test,mp_dev,direction):
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
	for i in range(len(MP)):
		if direction=="down":
			MP[i]=MP[i]-mp_dev
		if direction=="up":
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
	Models=["ExtraTrees"]
	Metrics=["ExtraTrees",mp_dev,direction,R2_2[0],RMSE[0],N1[0],N05[0]]
	return(Metrics,tree3preds.tolist())
	
def get_mets_preds(Dataset_name,ouput_preds,output_metrics):
	##master lists
	master_metrics=[]
	master_preds=[]
	##record column names for predictions as we go along
	col_names=["StdInChIKey"]
	##for each direction
	for direc in ["up","down"]:
		##for each deviation amount
		for dev in dev_list:
			##record deviation direction and amount
			col_names.append(str(dev) + " Deg " + direc)
			##reload dataset
			Dataset=pd.read_csv(os.path.join(dir,Dataset_name))
			##get train and test sets
			train=Dataset[Dataset["Train_test"]=="Train"]
			test=Dataset[Dataset["Train_test"]=="Test"]
			##get metrics and predictions and add to lists
			mets,preds=stat_split_metrics(train,test,dev,direc)
			master_metrics.append(mets)
			master_preds.append(preds)
	##gte inot dataframes and save
	master_metrics=pd.DataFrame(data=master_metrics,columns=["Model","Deviation","Direction","R2","RMSE","% Within 1","% Within 0.7"])
	master_preds.insert(0, test["StdInChIKey"].tolist())
	master_preds=pd.DataFrame(data=np.transpose(np.array(master_preds)),columns=col_names)
	master_preds.to_csv(output_preds,index=False)
	master_metrics.to_csv(output_metrics,index=False)

##section 4: run methods and save outputs
get_mets_preds(Dataset_name,output_preds,output_metrics)

