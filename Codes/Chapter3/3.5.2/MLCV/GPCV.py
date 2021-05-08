'''
This python script performs 10-fold cross validation for a dataset for GP machine learning method and outputs the resulting average metrics.
The metrics are calculated for each of the 10 folds then the mean taken as the final metrics.
In addition to the predictions, an upper and lower prediction (or error) is also calculated based on the prediction that encompasses 1 SD.
INPUTS:
Dataset.csv - a .csv file with the data in. The columns must be named in the same way as below
OUTPUTS:
GPCV_metrics - a .csv file of the metrics for GP method for its performance using 10-fold CV. "Max % within" refers to whether the prediction, with the upper and lower levels, fall within the range.
'''
#section 1: import modules
from sklearn.model_selection import KFold
import statistics
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
output_metrics=os.path.join(dir,"GPCV_metrics.csv")#location of output file for GPCV metrics

##section 3: define methods
#Define statistical measures and R2 conversion
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
def CV_metrics(Data,folds):
    #initiate lists to add metrics to
    RMSE=[]
    R2=[]
    N1=[]
    N05=[]
    N1_e=[]
    N05_e=[]
    GPR_RMSE=[]
    GPR_R2=[]
    GPR_N1=[]
    GPR_N05=[]
    GPR_N1_e=[]
    GPR_N05_e=[]
    #import Data
    X = Data
    X = X.sample(frac=1).reset_index(drop=True)
    #define k-fold cross validation
    col_names=X.dtypes.index
    X = np.array(X)
    kf = KFold(n_splits=folds)
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
        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        y_train=[[i] for i in y_train]
        y_train=np.array(y_train)
        #run models
        kernel = GPy.kern.RBF(input_dim=14, variance=1., lengthscale=1.)#####changing these initial parameters does not change the optimised ones
        GPR=GPy.models.GPRegression(X_train,y_train,kernel)
        GPR.optimize()
        gpr2preds = GPR.predict(X_test)[0]
        errors=GPR.predict_quantiles(X_test,quantiles=(16,84))########1 SD confidence interval
        gpr2preds=[i[0] for i in gpr2preds]
        errors[0]=[i[0] for i in errors[0]]
        errors[1]=[i[0] for i in errors[1]]
        #evaluate model
        GPR_R2.append(pearsonr(gpr2preds, y_test))
        GPR_RMSE.append(rmse(gpr2preds, y_test))
        GPR_N1.append(within_range(y_test,gpr2preds,1))
        GPR_N05.append(within_range(y_test,gpr2preds,0.7))
        GPR_N1_e.append(within_range_errors(y_test,gpr2preds,errors[0],errors[1],1))
        GPR_N05_e.append(within_range_errors(y_test,gpr2preds,errors[0],errors[1],0.7))
    #get R2 from Pearson output
    GPR_R2=get_R2(GPR_R2)
    #get mean metrics and put together in lists
    R2.append(statistics.mean(GPR_R2))
    RMSE.append(statistics.mean(GPR_RMSE))
    N1.append(statistics.mean(GPR_N1))
    N05.append(statistics.mean(GPR_N05))
    N1_e.append(statistics.mean(GPR_N1))
    N05_e.append(statistics.mean(GPR_N05))
    #
    #create dataframe of metrics
    Models=["GPR"]
    Metrics=list(zip(Models,R2,RMSE,N1,N05,N1_e,N05_e))
    Metrics_df=pd.DataFrame(data=Metrics, columns=['Model','R2','RMSE','% within 1','% within 0.7','Max % within 1','Max % within 0.7'])
    return(Metrics_df)
##method to put it all together
def get_CV_metrics(Dataset,output_metrics):
	##get metrics
	CV_metrics2=CV_metrics(Dataset,10)##10-folds
	##save to file
	CV_metrics2.to_csv(output_metrics,index=False)

##section 4: run CV method and get metrics
get_CV_metrics(Dataset,output_metrics)