'''
This python script generates the morgan fingerprint for each molecule and then finds the similarity between each molecule and its nearest neighbour (most similar neighbour)
INPUTS:
data1.csv - .csv file with a column for SMILES
data2.csv - .csv file with a column for SMILES
similarity.csv - directory and filename for results
OUTPUTS:
similarity.csv - file containing the similarity (from 1 to 0) of the nearest neighbour for each molecule
'''

##section 1: import modules
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AllChem import GetMorganFingerprint
from rdkit import Chem
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

##section 2: define inputs and outputs
##first dataset
data=pd.read_csv("data1.csv")
##second dataset
data2=pd.read_csv("data2.csv")
##output file name and location
output_file="similarity.csv"

##section3: methods to get fingerprints and nearest neighbours
##method to return fingerprints
def getFPs(smi_list):
    ##create molecule for every smiles
    ms=[Chem.MolFromSmiles(x) for x in smi_list]
    ##get fingerprint for every smiles
    fps = [GetMorganFingerprint(x,2) for x in ms]
    return(fps)
##method to return distance of nearest neighbour within same dataset
def getDist_same(fps):
    most_sim=[]
    for f in range(len(fps)):
        sims=[]
        for g in range(len(fps)):
            ##ignore if the same molecule!
            if f==g:
                continue
            sims.append(DataStructs.DiceSimilarity(fps[f],fps[g]))
        most_sim.append(max(sims))
    return(most_sim)
##method to return distance of nearest neighbour
def getDist_diff(fps1,fps2):
    most_sim=[]
    for f in range(len(fps1)):
        sims=[]
        for g in range(len(fps2)):
            sims.append(DataStructs.DiceSimilarity(fps1[f],fps2[g]))
        most_sim.append(max(sims))
    return(most_sim)

##section 4: runs methods and save output
sim_list=[]
##get SMILES
fps=getFPs(data["SMILES"])
fps2=getFPs(data2["SMILES"])
##get the similarity within the same dataset
sim_list.append(getDist_same(fps))
##get the similarity between different datasets
sim_list.append(getDist_diff(fps,fps2))
##save data
sim_list=pd.DataFrame(data=np.transpose(np.array(sim_list)),columns=["Data1 to Data1","Data1 to Data2"])
sim_list.to_csv(output_file,index=False)