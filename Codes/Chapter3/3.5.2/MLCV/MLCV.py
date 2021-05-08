'''
This python script performs 10-fold cross validation for a dataset for 7 ML methods and outputs the resulting average metrics.
The metrics are calculated for each of the 10 folds then the mean taken as the final metrics.
INPUTS:
Dataset.csv - a .csv file with the data in. The columns must be named in the same way as below
OUTPUTS:
MLCV_metrics - a .csv file of the metrics for each ML method for its performance using 10-fold CV.
'''
#section 1: import modules
import sys,os,re
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.cross_decomposition import PLSRegression
from sklearn import ensemble
from scipy.stats import pearsonr
import math
from sklearn.model_selection import KFold
import statistics

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of descriptor file
output_metrics=os.path.join(dir,"MLCV_metrics.csv")#location of output file for MLCV metrics

##section 3: define methods
#Define statistical measures and R2 conversion
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
#define method to get CV results
def CV_metrics(Data,folds,C,E,G):
	#initiate lists to add metrics to (one for )
	RMSE=[]
	R2=[]
	N1=[]
	N05=[]
	MLR_RMSE=[]
	MLR_R2=[]
	MLR_N1=[]
	MLR_N05=[]
	ANN_RMSE=[]
	ANN_R2=[]
	ANN_N1=[]
	ANN_N05=[]
	SVM_RMSE=[]
	SVM_R2=[]
	SVM_N1=[]
	SVM_N05=[]
	PLS_RMSE=[]
	PLS_R2=[]
	PLS_N1=[]
	PLS_N05=[]
	RF_RMSE=[]
	RF_R2=[]
	RF_N1=[]
	RF_N05=[]
	ET_RMSE=[]
	ET_R2=[]
	ET_N1=[]
	ET_N05=[]
	BG_RMSE=[]
	BG_R2=[]
	BG_N1=[]
	BG_N05=[]
	#import Data and randomise
	X = Data
	X = X.sample(frac=1).reset_index(drop=True)
	#define k-fold cross validation and make k splits
	col_names=X.dtypes.index
	X = np.array(X)
	kf = KFold(n_splits=folds)
	#for every split
	for train1, test1 in kf.split(X):
		train=X[train1]
		test=X[test1]
		train=pd.DataFrame(data=train, columns=col_names)
		test=pd.DataFrame(data=test, columns=col_names)
		X_train = train[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
					 'LsoluHsolv','LsolvHsolu','SASA','O_charges',
					 'C_charges','Most_neg','Most_pos','Het_charges']]
		y_train = train['LogS']
		X_test = test[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
					 'LsoluHsolv','LsolvHsolu','SASA','O_charges',
					 'C_charges','Most_neg','Most_pos','Het_charges']]
		y_test = test['LogS']
		y_test=np.array(y_test)
		scaler = preprocessing.StandardScaler().fit(X_train)
		X_train = scaler.transform(X_train)
		X_test = scaler.transform(X_test)
		#run models
		#MLR
		mlr = LinearRegression()
		mlr.fit(X_train, y_train)
		mlr2preds = mlr.predict(X_test)
		#evaluate model
		MLR_R2.append(pearsonr(mlr2preds, y_test))
		MLR_RMSE.append(rmse(mlr2preds, y_test))
		MLR_N1.append(within_range(y_test,mlr2preds,1))
		MLR_N05.append(within_range(y_test,mlr2preds,0.7))
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
		ANN_R2.append(pearsonr(mlp2preds, y_test))
		ANN_RMSE.append(rmse(mlp2preds, y_test))
		ANN_N1.append(within_range(y_test,mlp2preds,1))
		ANN_N05.append(within_range(y_test,mlp2preds,0.7))
		#SVM
		svm2 = svm.SVR(C = C, epsilon = E, gamma = G, kernel = 'rbf')
		svm2.fit(X_train, y_train)
		svm2preds = svm2.predict(X_test)
		#evaluate model
		SVM_R2.append(pearsonr(svm2preds, y_test))
		SVM_RMSE.append(rmse(svm2preds, y_test))
		SVM_N1.append(within_range(y_test,svm2preds,1))
		SVM_N05.append(within_range(y_test,svm2preds,0.7))
		#PLS
		pls2 = PLSRegression(n_components=9)
		pls2.fit(X_train, y_train)
		pls2preds = pls2.predict(X_test)
		#evaluate model
		#convert to float (comes in weird type?)
		pls2preds2=[]
		for i in pls2preds:
			pls2preds2.append(float(i))
		PLS_R2.append(pearsonr(pls2preds2, y_test))
		PLS_RMSE.append(rmse(pls2preds2, y_test))
		PLS_N1.append(within_range(y_test,pls2preds2,1))
		PLS_N05.append(within_range(y_test,pls2preds2,0.7))
		#RF
		tree2 = ensemble.RandomForestRegressor(n_estimators=500,n_jobs=-1)
		tree2.fit(X_train, y_train)
		tree2preds = tree2.predict(X_test)
		#evaluate model
		RF_R2.append(pearsonr(tree2preds, y_test))
		RF_RMSE.append(rmse(tree2preds, y_test))
		RF_N1.append(within_range(y_test,tree2preds,1))
		RF_N05.append(within_range(y_test,tree2preds,0.7))
		#ExtraTrees
		tree3 = ensemble.ExtraTreesRegressor(n_estimators=500,n_jobs=-1)
		tree3.fit(X_train, y_train)
		tree3preds = tree3.predict(X_test)
		#evaluate model
		ET_R2.append(pearsonr(tree3preds, y_test))
		ET_RMSE.append(rmse(tree3preds, y_test))
		ET_N1.append(within_range(y_test,tree3preds,1))
		ET_N05.append(within_range(y_test,tree3preds,0.7))
		#Bagging
		tree4 = ensemble.BaggingRegressor(n_estimators=500,n_jobs=-1)
		tree4.fit(X_train, y_train)
		tree4preds = tree4.predict(X_test)
		#evaluate model
		BG_R2.append(pearsonr(tree4preds, y_test))
		BG_RMSE.append(rmse(tree4preds, y_test))
		BG_N1.append(within_range(y_test,tree4preds,1))
		BG_N05.append(within_range(y_test,tree4preds,0.7))
	#get R2 from Pearson output
	MLR_R2=get_R2(MLR_R2)
	ANN_R2=get_R2(ANN_R2)
	SVM_R2=get_R2(SVM_R2)
	PLS_R2=get_R2(PLS_R2)
	RF_R2=get_R2(RF_R2)
	ET_R2=get_R2(ET_R2)
	BG_R2=get_R2(BG_R2)
	#get mean metrics and put together in lists
	R2.append(statistics.mean(MLR_R2))
	RMSE.append(statistics.mean(MLR_RMSE))
	N1.append(statistics.mean(MLR_N1))
	N05.append(statistics.mean(MLR_N05))
	#
	R2.append(statistics.mean(ANN_R2))
	RMSE.append(statistics.mean(ANN_RMSE))
	N1.append(statistics.mean(ANN_N1))
	N05.append(statistics.mean(ANN_N05))
	#
	R2.append(statistics.mean(SVM_R2))
	RMSE.append(statistics.mean(SVM_RMSE))
	N1.append(statistics.mean(SVM_N1))
	N05.append(statistics.mean(SVM_N05))
	#
	R2.append(statistics.mean(PLS_R2))
	RMSE.append(statistics.mean(PLS_RMSE))
	N1.append(statistics.mean(PLS_N1))
	N05.append(statistics.mean(PLS_N05))
	#
	R2.append(statistics.mean(RF_R2))
	RMSE.append(statistics.mean(RF_RMSE))
	N1.append(statistics.mean(RF_N1))
	N05.append(statistics.mean(RF_N05))
	#
	R2.append(statistics.mean(ET_R2))
	RMSE.append(statistics.mean(ET_RMSE))
	N1.append(statistics.mean(ET_N1))
	N05.append(statistics.mean(ET_N05))
	#
	R2.append(statistics.mean(BG_R2))
	RMSE.append(statistics.mean(BG_RMSE))
	N1.append(statistics.mean(BG_N1))
	N05.append(statistics.mean(BG_N05))
	#
	#create dataframe of metrics
	Models=["MLR","ANN","SVM","PLS","RF","ExtraTrees","Bagging"]
	Metrics=list(zip(Models,R2,RMSE,N1,N05))
	Metrics_df=pd.DataFrame(data=Metrics, columns=['Model','R2','RMSE','% within 1','% within 0.7'])
	return(Metrics_df)
##method to put it all together
def get_CV_metrics(Dataset,output_metrics):
	##get metrics
	CV_metrics2=CV_metrics(Dataset,10,4,0.01,0.03)##10-folds and SVM parameters
	##save to file
	CV_metrics2.to_csv(output_metrics,index=False)

##section 4: run CV method and get metrics
get_CV_metrics(Dataset,output_metrics)