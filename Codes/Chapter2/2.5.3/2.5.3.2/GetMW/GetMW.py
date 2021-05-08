'''
This python script uses CIRpy to get molecule weight from SMILES strings for a dataset.
INPUTS:
dataset.csv - dataset with a column called "SMILES" with SMILES codes and a column called "Name" containing a unique identifier
OUTPUTS:
dataset_MW.csv - new file containing MW additionally
'''
##section 1: import modules
import cirpy,os
import pandas as pd

##section 2: define input and output files
dir=os.path.dirname(__file__)#get current directory to join to files
dataset=pd.read_csv(os.path.join(dir,"dataset.csv"))#location of dataset file
output=os.path.join(dir,"dataset_MW.csv")#location of output file

##section 3: method for getting MW
def get_MW(input_file,output_file):
	##get list of SMILES
	smi=input_file["SMILES"]
	##get list to put MWs in
	smi_list=[]
	##get MW for every SMILES
	for i in smi:
		smi_list.append(cirpy.resolve(i, 'mw'))
	##put in new column for MW
	input_file["MW"]=smi_list
	##save to file
	input_file.to_csv(output_file,index=False)

##section 4: run method and save output
get_MW(dataset,output)

