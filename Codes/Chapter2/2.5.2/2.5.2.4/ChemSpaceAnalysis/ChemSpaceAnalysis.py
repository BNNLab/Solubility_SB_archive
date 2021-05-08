'''
This script takes a dataset and calculates the Morgan fingerprint
PCA analysis is performed and the principal components outputted (this can then be plotted) or analysed further.
INPUTS:
dataset.csv - dataset with a column called "SMILES" with SMILES codes and a column called "Name" containing a unique identifier
OUTPUTS:
dataset_PCA.csv - dataset containing the first two principal components calculated from the Morgan fingerprints
dataset_scree.csv - The cumulative variance described by first two principal components (known as a "scree" plot when plotted)
'''
##section 1: import modules
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import os

##section 2: define input and output files
dir=os.path.dirname(__file__)#get current directory to join to files
dataset=pd.read_csv(os.path.join(dir,"dataset.csv"))#location of dataset file
output_PCA=os.path.join(dir,"dataset_PCA.csv")#location of output file for PCA
output_scree=os.path.join(dir,"dataset_scree.csv")#location of output file for PCA

##section 3: define methods
##method to convert fingerprint into an array you can perform PCA on
def fp2arr(fp):
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp,arr)
    return arr
##method to get PCA and scree data
def get_PCA(data,output_PCA,output_scree):
	##set up Morgan Fingerprint, radius 2
	fpgen = rdFingerprintGenerator.GetMorganGenerator(2)
	##get a molecule object for every molecule
	mols = [Chem.MolFromSmiles(smi) for smi in data["SMILES"].tolist()]
	##get a 2D structure for each molecule
	for m in mols:
		AllChem.Compute2DCoords(m)
	##get fingerprint for every molecule
	fps = [fpgen.GetFingerprint(m) for m in mols]
	##convert to array
	X = np.asarray([fp2arr(fp) for fp in fps])
	##set up PCA
	pca = PCA(n_components=2)
	##perform PCA
	res = pca.fit_transform(X)
	##get cumulative variance explained by the first two components and make dataframe
	cum_scree=np.cumsum(pca.explained_variance_ratio_)*100
	cum_scree=pd.DataFrame(data=[cum_scree],columns=["Cumulative Variance PCA 1","Cumulative Variance PCA 2"])
	##transform into PCA dataframe
	df_PCA=pd.DataFrame(data=res,columns=["PCA1","PCA2"])
	##insert name and smiles
	df_PCA["Name"]=data["Name"].tolist()
	df_PCA["SMILES"]=data["SMILES"].tolist()
	##reorder dataframe
	df_PCA=df_PCA[["Name","SMILES","PCA1","PCA2"]]
	##output PCA dataframe
	df_PCA.to_csv(output_PCA,index=False)
	##output cumulative variance
	cum_scree.to_csv(output_scree,index=False)

##section 4: run method and save output
get_PCA(dataset,output_PCA,output_scree)