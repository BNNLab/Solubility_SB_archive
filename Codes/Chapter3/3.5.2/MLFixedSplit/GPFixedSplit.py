'''
This python script performs gaussian process regression for a dataset.
The model is built woth training data and the metrics and individual predictions on test data is outputted.
In addition to the predictions, an upper and lower prediction (or error) is also calculated based on the prediction that encompasses 1 SD.
INPUTS:
Dataset.csv - a .csv file with the data in. The columns must be named in the same way as below
OUTPUTS:
ML_preds.csv - a .csv file containing the individual predictions for the test set for all 7 ML methods. "Upper" and "Lower" refer to the range encompasses by 1 SD (error).
ML_metrics - a .csv file of the metrics for each ML method for its performance on the test sets. "Max % within" refers to whether the prediction, with the upper and lower levels, fall within the range.
'''
##section 1: import modules
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
import GPy
from scipy.stats import pearsonr
import math

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of descriptor file
output_preds=os.path.join(dir,"GP_preds.csv")#location of output file for ML predictions
output_metrics=os.path.join(dir,"GP_metrics.csv")#location of output file for ML metrics

##section 3: define methods for getting ML results
##split into train and test (train on the former, test on the latter)
train=Dataset[Dataset["Train_test"]=="Train"]
test=Dataset[Dataset["Train_test"]=="Test"]
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
def within_range_errors(list1, list2, list3, list4, range2):
	x=0
	for i in range(len(list2)):
		if (list1[i]-range2)<= list2[i] <= (list1[i]+range2): 
			x+=1
		elif (list1[i]-range2)<= list3[i] <= (list1[i]+range2): 
			x+=1
		elif (list1[i]-range2)<= list4[i] <= (list1[i]+range2): 
			x+=1
	return((float(x)/(len(list2)))*100)
#fixed split method to get predictions and metrics
def stat_split_metrics(train,test):
	##list for metrics
	RMSE=[]
	R2=[]
	N1=[]
	N05=[]
	N1_e=[]
	N05_e=[]
	#place target value in y
	y_train = train['LogS']
	y_test = test['LogS']
	y_test = np.array(y_test)
	#place descriptors in X
	X_train = train[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
					 'LsoluHsolv','LsolvHsolu','SASA','O_charges',
					 'C_charges','Most_neg','Most_pos','Het_charges']]
	X_test = test[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
					 'LsoluHsolv','LsolvHsolu','SASA','O_charges',
					 'C_charges','Most_neg','Most_pos','Het_charges']]
	#scale data
	scaler = preprocessing.StandardScaler().fit(X_train)
	X_train = scaler.transform(X_train)
	X_test = scaler.transform(X_test)
	##everything has to be in individual lists
	y_train=[[i] for i in y_train]
	y_train=np.array(y_train)
	#run models
	#GPR
	kernel = GPy.kern.RBF(input_dim=14, variance=1., lengthscale=1.)#####changing these initial parameters does not change the optimised ones
	GPR=GPy.models.GPRegression(X_train,y_train,kernel)
	GPR.optimize()
	gpr2preds = GPR.predict(X_test)[0]
	errors=GPR.predict_quantiles(X_test,quantiles=(16,84))########1 SD confidence interval
	gpr2preds=[i[0] for i in gpr2preds]##predictions come as lists
	errors[0]=[i[0] for i in errors[0]]##get the upper and lower limits as errors
	errors[1]=[i[0] for i in errors[1]]
	#evaluate model
	R2.append(pearsonr(gpr2preds, y_test))
	RMSE.append(rmse(gpr2preds, y_test))
	N1.append(within_range(y_test,gpr2preds,1))
	N05.append(within_range(y_test,gpr2preds,0.7))
	##these are whether the errors put the prediction in range
	N1_e.append(within_range_errors(y_test,gpr2preds,errors[0],errors[1],1))
	N05_e.append(within_range_errors(y_test,gpr2preds,errors[0],errors[1],0.7))
	#get R2 from Pearson output
	R2_2=[]
	for i in range(len(R2)):
		x=re.findall('\d\.\d+',str(R2[i]))
		j=float(x[0])
		j=j**2
		R2_2.append(j)
	#create dataframe of metrics
	Models=["GPR"]
	Metrics=list(zip(Models,R2_2,RMSE,N1,N05,N1_e,N05_e))
	Metrics_df=pd.DataFrame(data=Metrics, columns=['Model','R2','RMSE','% within 1','% within 0.7','Max % within 1','Max % within 0.7'])
	#create a dataframe of predictions
	indiv_preds=list(zip(test['StdInChIKey'],gpr2preds,errors[0],errors[1],y_test))
	indiv_preds_df=pd.DataFrame(data=indiv_preds, columns=["StdInChIKey","GPR","Lower","Upper","Experimental"])
	return(Metrics_df,indiv_preds_df)
##method to put it all together
def get_preds_metrics(train,test,output_metrics,output_preds):
	##get metrics and predictions
	metrics,preds=stat_split_metrics(train,test)##these are the SVM parameters previously determined
	##save results to file
	metrics.to_csv(output_metrics,index=False)
	preds.to_csv(output_preds,index=False)

##section 4: run method to get predictions and metrics
get_preds_metrics(train,test,output_metrics,output_preds)