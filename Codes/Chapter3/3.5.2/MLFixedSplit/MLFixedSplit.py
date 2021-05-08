'''
This python script performs machine learning on a dataset and outputs the metrics of performance on a test set and the indivisual predictions.
MLR, PLS, ANN, SVM, ET, RF and Bagging are trained on the training data and tested on the test data.
Parameters were previously determined.
INPUTS:
Dataset.csv - a .csv file with the data in. The columns must be named in the same way as below
OUTPUTS:
ML_preds.csv - a .csv file containing the individual predictions for the test set for all 7 ML methods
ML_metrics - a .csv file of the metrics for each ML method for its performance on the test sets
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
from sklearn.linear_model import LinearRegression
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter('error', category=ConvergenceWarning)
warnings.simplefilter('ignore', category=DataConversionWarning)##these ignore convergence warnings for ANN

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of descriptor file
output_preds=os.path.join(dir,"ML_preds.csv")#location of output file for ML predictions
output_metrics=os.path.join(dir,"ML_metrics.csv")#location of output file for ML metrics

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
#fixed split method to get predictions and metrics
def stat_split_metrics(train,test,C,E,G):
	#lists for metrics
	RMSE=[]
	R2=[]
	N1=[]
	N05=[]
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
	#run models
	#MLR
	mlr = LinearRegression()
	mlr.fit(X_train, y_train)
	mlr2preds = mlr.predict(X_test)
	#evaluate model
	R2.append(pearsonr(mlr2preds, y_test))
	RMSE.append(rmse(mlr2preds, y_test))
	N1.append(within_range(y_test,mlr2preds,1))
	N05.append(within_range(y_test,mlr2preds,0.7))
	#ANN
	mlp = MLPRegressor(hidden_layer_sizes=300,max_iter=800)
	for f in range(100):
		try:
			mlp.fit(X_train, y_train)
			mlp2preds = mlp.predict(X_test)
			if np.ptp(mlp2preds) == 0:
				continue
			break
		except:
			continue
	#evaluate model
	R2.append(pearsonr(mlp2preds, y_test))
	RMSE.append(rmse(mlp2preds, y_test))
	N1.append(within_range(y_test,mlp2preds,1))
	N05.append(within_range(y_test,mlp2preds,0.7))
	#SVM
	svm2 = svm.SVR(C = C, epsilon = E, gamma = G, kernel = 'rbf')####found in grid search
	svm2.fit(X_train, y_train)
	svm2preds = svm2.predict(X_test)
	#evaluate model
	R2.append(pearsonr(svm2preds, y_test))
	RMSE.append(rmse(svm2preds, y_test))
	N1.append(within_range(y_test,svm2preds,1))
	N05.append(within_range(y_test,svm2preds,0.7))
	#PLS
	pls2 = PLSRegression(n_components=9)
	pls2.fit(X_train, y_train)
	pls2preds = pls2.predict(X_test)
	#evaluate model
	#convert to float (comes in weird type?)
	pls2preds2=[]
	for i in pls2preds:
		pls2preds2.append(float(i))
	R2.append(pearsonr(pls2preds2, y_test))
	RMSE.append(rmse(pls2preds2, y_test))
	N1.append(within_range(y_test,pls2preds2,1))
	N05.append(within_range(y_test,pls2preds2,0.7))
	#RF
	tree2 = ensemble.RandomForestRegressor(n_estimators=500,n_jobs=-1)
	tree2.fit(X_train, y_train)
	tree2preds = tree2.predict(X_test)
	#evaluate model
	R2.append(pearsonr(tree2preds, y_test))
	RMSE.append(rmse(tree2preds, y_test))
	N1.append(within_range(y_test,tree2preds,1))
	N05.append(within_range(y_test,tree2preds,0.7))
	#ExtraTrees
	tree3 = ensemble.ExtraTreesRegressor(n_estimators=500,n_jobs=-1)
	tree3.fit(X_train, y_train)
	tree3preds = tree3.predict(X_test)
	#evaluate model
	R2.append(pearsonr(tree3preds, y_test))
	RMSE.append(rmse(tree3preds, y_test))
	N1.append(within_range(y_test,tree3preds,1))
	N05.append(within_range(y_test,tree3preds,0.7))
	#Bagging
	tree4 = ensemble.BaggingRegressor(n_estimators=500,n_jobs=-1)
	tree4.fit(X_train, y_train)
	tree4preds = tree4.predict(X_test)
	#evaluate model
	R2.append(pearsonr(tree4preds, y_test))
	RMSE.append(rmse(tree4preds, y_test))
	N1.append(within_range(y_test,tree4preds,1))
	N05.append(within_range(y_test,tree4preds,0.7))
	#get R2 from Pearson output
	R2_2=[]
	for i in range(len(R2)):
		x=re.findall('\d\.\d+',str(R2[i]))
		j=float(x[0])
		j=j**2
		R2_2.append(j)
	#create dataframe of metrics
	Models=["MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"]
	Metrics=list(zip(Models,R2_2,RMSE,N1,N05))
	Metrics_df=pd.DataFrame(data=Metrics, columns=['Model','R2','RMSE','% within 1','% within 0.7'])
	#create dataframe of individual predictions
	indiv_preds=list(zip(test['StdInChIKey'],mlr2preds,mlp2preds,svm2preds,pls2preds2,tree2preds,tree3preds,tree4preds))
	indiv_preds_df=pd.DataFrame(data=indiv_preds, columns=["StdInChIKey","MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"])
	return(Metrics_df,indiv_preds_df)
##method to put it all together
def get_preds_metrics(train,test,output_metrics,output_preds):
	##get metrics and predictions
	metrics,preds=stat_split_metrics(train,test,4,0.1,0.03)##these are the SVM parameters previously determined
	##save results to file
	metrics.to_csv(output_metrics,index=False)
	preds.to_csv(output_preds,index=False)

##section 4: run method to get predictions and metrics
get_preds_metrics(train,test,output_metrics,output_preds)