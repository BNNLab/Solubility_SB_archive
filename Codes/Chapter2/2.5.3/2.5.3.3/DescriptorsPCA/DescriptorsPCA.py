'''
This python script performs PCA analysis on the 22 initial descriptors and outputs the principal components and cumulative variance for each successive component.
INPUTS:
Descriptors.csv - a .csv file containing calculated descriptors for a dataset, the descriptors must be named as in the global variable "Descs"
OUTPUTS:
Descriptors_PCA.csv - a .csv file the same as Descriptors.csv but also containing the PCA dataset
Descriptors_scree.csv - a .csv file containing the cumulative variance for each successive component (known as a scree plot when plotted)
'''
##section 1: import modules
from sklearn import preprocessing
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import os

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Descriptors=pd.read_csv(os.path.join(dir,"Descriptors.csv"))#location of descriptor file
output_PCA=os.path.join(dir,"Descriptors_PCA.csv")#location of output file for PCA
output_scree=os.path.join(dir,"Descriptors_scree.csv")#location of output file for PCA

##section 3: define PCA method
##name of new PCA columns
columns =['principal component 1', 'principal component 2','principal component 3','principal component 4',
'principal component 5','principal component 6','principal component 7','principal component 8',
'principal component 9','principal component 10','principal component 11','principal component 12',
'principal component 13','principal component 14','principal component 15','principal component 16',
'principal component 17','principal component 18','principal component 19','principal component 20',
'principal component 21','principal component 22']
##names of the descriptor columns in Descriptors.csv
Descs=['E0_gas','E0_solv','DeltaE0_sol','G_gas','G_solv','DeltaG_sol','HOMO','LUMO','LsoluHsolv','LsolvHsolu',
	   'gas_dip','solv_dip','O_charges','C_charges','Most_neg','Most_pos','Het_charges','Volume','SASA','MW','N_atoms','MP']
def get_PCA(Descriptors,output_PCA,ouput_scree):
	##get just the descriptors from data file
	Data=Descriptors[Descs]
	##scale data
	scaler = preprocessing.StandardScaler().fit(Data)
	Data = scaler.transform(Data)
	##set up PCA with n_comp=n_desc
	pca = PCA(n_components=22)
	##get components
	principalComponents = pca.fit_transform(Data)
	##make a dataframe with the PCAs in 
	principalDf = pd.DataFrame(data=principalComponents, columns=columns)
	##append new data to original data
	new_data=pd.concat([Descriptors,principalDf],axis=1)
	##save data
	new_data.to_csv(output_PCA,index=False)
	##get scree plot data (cumulative variance explained by components)
	cum_scree=np.cumsum(pca.explained_variance_ratio_)*100
	##make into dataframe and save
	cum_scree=pd.DataFrame(data=[cum_scree],columns=columns)
	cum_scree.to_csv(output_scree,index=False)

##section 4: run method and get PCA data
get_PCA(Descriptors,output_PCA,output_scree)