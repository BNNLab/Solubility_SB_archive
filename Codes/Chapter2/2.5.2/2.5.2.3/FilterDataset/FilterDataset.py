'''
This python script uses openbabel and SMARTS to remove data with disallowed elements (e.g. metals) and mixtures.
INPUTS:
Dataset.csv - a .csv containing a column with SMILES codes in called "SMILES"
OUTPUTS:
Dataset_trimmed.csv - a .csv file containing the data without disallowed molecules removed.
'''
##section 1: import modules
import re,periodictable,os
import pandas as pd
from pybel import *
import numpy as np

##section 2: define inputs and outputs
dir=os.path.dirname(__file__)#get current directory to join to files
Dataset=pd.read_csv(os.path.join(dir,"Dataset.csv"))#location of dataset file
output_dataset=os.path.join(dir,"Dataset_trimmed.csv")#location of output file for trimmed dataset


##section 3: define method to filter dataset
def trim_dataset(data,output):
	#create a list of elements to keep AND elements not supported by openbabel (cannot remove them anyway)
	els=["H","C","N","O","F","P","S","Cl","Br","I","Rf","Db","Sg","Bh","Hs","Mt","Ds","Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"]
	#get the inverse of this list, elements to discard
	elements=[]
	for el in periodictable.elements: 
		elements.append(el.symbol)
	del(elements[0])
	for f in els:
		elements.remove(f)
	#convert list of elements to list of SMARTS codes
	elements2=[]
	for f in elements:
		elements2.append("[" + f + "]")##"[El]" is the SMARTS code for detecting an element
	##get column names for re-making dataframe later
	columns=list(data)
	##find column number containing SMILES
	for col in range(len(columns)):
		if columns[col] == "SMILES":
			smi_col=col
	##convert to numpy array
	data=np.array(data)
	#only keep data where SMILES does not contain a dot (remove mixtures)
	keep_data=[]
	for f in range(len(data)):
		if re.findall("\.",data[f][smi_col])==[]:##points to column containing SMILES
			keep_data.append(data[f])
	#create a list of data to discard (contains one or more of the disallowed elements)
	discard=[]
	for f in range(len(keep_data)):
		mol=readstring("smi",keep_data[f][smi_col])
		#match all SMARTS to smile
		for g in elements2:
			smarts=Smarts(g)
			if smarts.findall(mol) != []:
				discard.append(keep_data[f])
				continue
	#get a full list of SMILES and a discard list
	data_new=[]
	smi1=[]
	smi2=[]
	for f in keep_data:
		smi1.append(f[smi_col])
	for f in discard:
		smi2.append(f[smi_col])
	#filter out discard molecules
	smi_unique=[]
	for f in smi1:
		if f in smi2:
			continue
		smi_unique.append(f)
	for f in smi_unique:
		for g in range(len(keep_data)):
			if keep_data[g][smi_col] == f:
				data_new.append(keep_data[g])
	#make new dataframe and save
	df=pd.DataFrame(data=data_new,columns=columns)
	df.to_csv(output,index=False)

##section 4: run method and save filtered dataset
trim_dataset(Dataset,output_dataset)
