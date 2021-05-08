'''
This script finds the correlation matrix for the descriptors.
INPUTS:
Descriptors.csv - a .csv file containing calculated descriptors for a dataset, the descriptors must be named as in the global variable "Descs"
OUTPUTS:
Descriptors_corr.csv - a .csv containing the correlation matrix
'''
##section 1: import modules
import pandas as pd
import os

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Descriptors=pd.read_csv(os.path.join(dir,"Descriptors.csv"))#location of descriptor file
output_corr=os.path.join(dir,"Descriptors_corr.csv")#location of output file for correlation matrix

##section 3: define correlation method
##names of the descriptor columns in Descriptors.csv
Descs=['E0_gas','E0_solv','DeltaE0_sol','G_gas','G_solv','DeltaG_sol','HOMO','LUMO','LsoluHsolv','LsolvHsolu',
	   'gas_dip','solv_dip','O_charges','C_charges','Most_neg','Most_pos','Het_charges','Volume','SASA','MW','N_atoms','MP']
def get_corr(Data,output):
	##get just descriptors
	Data=Data[Descs]
    ##get correlation matrix
	corr=Data.corr()
    ##convert to R-squared
	corr=corr.pow(2)
	##save output
	corr.to_csv(output)

##section 4: run method and save output
get_corr(Descriptors,output_corr)