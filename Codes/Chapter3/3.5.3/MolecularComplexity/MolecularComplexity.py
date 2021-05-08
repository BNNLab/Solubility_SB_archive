'''
This python script computes the BertzCT complexity index for a dataset via rdkit.
INPUTS:
Dataset.csv - a .csv file with a column called "SMILES" containing the SMILES codes for the molecules to be analysed
OUTPUTS:
Dataset_complexity - a .csv the same as Dataset.csv but with an extra column called "Complexity" containing the complexity index for each molecule
'''
##section 1: import modules
from rdkit import Chem
import pandas as pd
from rdkit.Chem.GraphDescriptors import BertzCT
import os

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of Dataset file
output_complexity=os.path.join(dir,"Dataset_complexity.csv")#location of output file for molecular complexity

##section 3: define method to get complexity
def get_compl(Data,Output):
	##get list of SMILES
	smis=Data["SMILES"].tolist()
	##new list
	compl=[]
	##for every molecule
	for f in smis:
		##make molecular object
		m=Chem.MolFromSmiles(f)
		##add hydrogens 
		m=Chem.AddHs(m)
		##get complexity
		compl.append(BertzCT(m))
	##add new column on end of data file with complexity and save
	Data["Complexity"]=compl
	Data.to_csv(Output,index=False)

##section 4: run method to save complexity
get_compl(Dataset,output_complexity)