'''
This python script generates up to 20 conformers for each molecule using rdkit.
INPUTS:
input_file - .csv files with a column with StdInChIKey and a column with SMILES
output_paths - directory to save conformation structures
OUTPUTS:
output_paths - Directory containing a directory for each molecule. These directories contain the structures of up to 20 conformers.
'''

##section 1: import modules
import os
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

##section 2: define inputs and outputs
input_file="mol_smi.csv"
inputs=pd.read_csv(input_file)
output_paths="Confs"
os.makedirs(output_paths)

##section 3: get and save conformers for each molecule
##list of SMILES
smis=inputs["SMILES"]
##list of StdInChIKeys, unique identifiers for making a folder for each molecule
stds=inputs["StdInChIKey"]
##for every molecule
for f in range(len(smis)):
    ##generate mol object
    m=Chem.MolFromSmiles(smis[f])
    ##make folder for molecule
    path=output_paths + "\\" + stds[f]
    os.mkdir(path)
    ##add hydrogens
    m=Chem.AddHs(m)
    ##keep chiral
    Chem.AssignAtomChiralTagsFromStructure(m, replaceExistingTags=True)
    ##number of conformers
    nc=20
    ##get conformers
    AllChem.EmbedMultipleConfs(m,nc,pruneRmsThresh=0.5)
    ##optimsie with MM
    _=AllChem.MMFFOptimizeMoleculeConfs(m,maxIters=1000)
    ##arbitrary conformer number counter for file names
    i = 0
    for conf in m.GetConformers():
        ##procedure to get and save sdf structures
        tm = Chem.Mol(m,False,conf.GetId())
        file = path + "\\out" + str(i)+".sdf"
        writer = Chem.SDWriter(file) #writes sdf file for each confomer
        writer.write(tm)
        writer.close()
        i+=1