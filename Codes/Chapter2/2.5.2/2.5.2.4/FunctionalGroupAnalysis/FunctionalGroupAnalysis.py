'''
This script uses the openbabel python module to detect predefined functional groups in molecules.
SMARTS, defined in a separate file, are matched to the SMILES string of the molecule.
INPUTS:
SMARTS.csv - a .csv file with column "FG" with names of functional groups and column "SMARTS" with the SMARTS codes for the functional groups.
dataset.csv - a .csv file that must contain a column called "SMILES" containing the SMILES codes of the molecule to be analysed and a column called "Name" containing a unique identifier for that molecule
OUTPUTS:
dataset_FG.csv - a .csv file with a new column for each functional group. If functional group present, "1" appended, otherwise "0" appended.
'''
##section 1: import modules
import openbabel
import csv,os
from pybel import *
import pandas as pd
import numpy as np

##section 2: define input and output files
dir=os.path.dirname(__file__)#get current directory to join to files
SMARTS=pd.read_csv(os.path.join(dir,"SMARTS.csv"))#location of SMARTS file
dataset=pd.read_csv(os.path.join(dir,"dataset.csv"))#location of dataset file
output=os.path.join(dir,"dataset_FG.csv")#location of output file

##section 3: define FG method
def get_FG(SMARTS,dataset,output):
	##Functional Group names
	FG_names=SMARTS['FG']
	##Get SMART CODES
	SMARTS_codes=SMARTS['SMARTS']
	##Get SMILES
	smiles=dataset['SMILES']
	##Get unique identifier
	name=dataset['Name']
	##convert from pandas dataframe to numpy array
	#water1=np.array(water1)
	##where the FG matches will be recorded
	FG_list=[]
	for f in range(len(dataset)):##for every molecule
		FG=[]##each new row of output
		##Get unique identifier
		FG.append(name[f])
		##Get SMILES
		FG.append(smiles[f])
		##Create molecule object from smiles
		mol=readstring("smi",smiles[f])
		##match all SMARTS to SMILES, append "0" if no match, otherwise append "1"
		for g in SMARTS_codes:
			smarts=Smarts(g)
			if smarts.findall(mol) == []:
				FG.append("0")
			else:
				FG.append("1")
		##add to master list
		FG_list.append(FG)
	##get list of functional group names
	FG_names2=list(FG_names)
	##insert SMILES in front of list of functional groups
	FG_names2.insert(0,"SMILES")
	##insert name in front of list of functional groups
	FG_names2.insert(0,"Name")
	##make a pandas dataframe
	FG_list=pd.DataFrame(data=FG_list,columns=FG_names2)
	##save output
	FG_list.to_csv(output,index=False)

##section 4: run method and save output
get_FG(SMARTS,dataset,output)