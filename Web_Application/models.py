#import modules required
import __main__
__main__.pymol_argv = ['pymol','-qc'] # Pymol: quiet and no GUI
import pymol,sys,os,re
pymol.finish_launching()
import sys,os,re,pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.neural_network import MLPRegressor
from sklearn import svm
from sklearn import ensemble
import math
from joblib import dump, load
import GPy
from molmod import *

##get training sets
water_loc1_train="model_data/water/water1_gen1_train.csv"
ethanol_loc_train="model_data/ethanol/ethanol_gen1_train.csv"
benzene_loc_train="model_data/benzene/benzene_gen1_train.csv"
acetone_loc_train="model_data/acetone/acetone_gen1_train.csv"
water_train=pd.read_csv(water_loc1_train)
ethanol_train=pd.read_csv(ethanol_loc_train)
benzene_train=pd.read_csv(benzene_loc_train)
acetone_train=pd.read_csv(acetone_loc_train)


##step 2 call back models and make prediction WITH MP
def get_pred(model,train,test):
    #place target value in y
    y_train = train['LogS']
    #place descriptors in X
    X_train = train[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges']]
    X_test = test[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges']]
    #scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    ##load model, get preds
    reg = load(model)
    reg2preds = reg.predict(X_test)
    return(reg2preds[0])


def get_pred_GPR(model,train,test):
    #place target value in y
    y_train = train['LogS']
    #place descriptors in X
    X_train = train[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges']]
    X_test = test[['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges']]
    #scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    ##load model, get preds
    GPR=pickle.load(open(model,"rb"))
    gpr2preds = GPR.predict(X_test)[0]
    errors=GPR.predict_quantiles(X_test,quantiles=(16,84))########1 SD confidence interval
    gpr2preds=[i[0] for i in gpr2preds]
    errors[0]=[i[0] for i in errors[0]]
    errors[1]=[i[0] for i in errors[1]]
    return(gpr2preds[0],errors[0][0],errors[1][0])


def get_preds_data(test,train,solvent):    
    preds=[]
    preds.append(get_pred("model_data/" + solvent + "/ET_MP.dump",train,test))
    preds.append(get_pred("model_data/" + solvent + "/SVM_MP.dump",train,test))
    preds.append(get_pred("model_data/" + solvent + "/MLP_MP.dump",train,test))
    gpr_preds,lower,upper=get_pred_GPR("model_data/" + solvent + "/GPR_MP.dump",train,test)
    preds.append(gpr_preds)
    mean=np.mean(preds)
    median=np.median(preds)
    preds.append(upper)
    preds.append(lower)
    preds.append(mean)
    preds.append(median)
    preds=[["ET","SVM","ANN","GP","GP_Upper","GP_Lower","Consensus_Mean","Consensus_Median"],["ExtraTrees Prediction","Support Vector Machine Prediction","Artificial Neural Networks Prediction","Gaussian Process Regression Prediction","Upper limit of GPR prediction (error to 1 SD)","Lower limit of GPR prediction (error to 1 SD)","Mean prediction from ET, SVM, ANN and GPR","Median prediction from ET, SVM, ANN and GPR"],preds]
    preds=np.array(preds).T.tolist()
    preds=pd.DataFrame(data=preds,columns=["Method","Algorithm Explanation","Prediction (LogS)"])
    return(preds)

##step 2 call back models and make prediction NO MP
def get_pred_noMP(model,train,test):
    #place target value in y
    y_train = train['LogS']
    #place descriptors in X
    X_train = train[['MW','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges']]
    X_test = test[['MW','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges']]
    #scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    ##load model, get preds
    reg = load(model)
    reg2preds = reg.predict(X_test)
    return(reg2preds[0])


def get_pred_GPR_noMP(model,train,test):
    #place target value in y
    y_train = train['LogS']
    #place descriptors in X
    X_train = train[['MW','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges']]
    X_test = test[['MW','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges']]
    #scale data
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_test = scaler.transform(X_test)
    ##load model, get preds
    GPR=pickle.load(open(model,"rb"))
    gpr2preds = GPR.predict(X_test)[0]
    errors=GPR.predict_quantiles(X_test,quantiles=(16,84))########1 SD confidence interval
    gpr2preds=[i[0] for i in gpr2preds]
    errors[0]=[i[0] for i in errors[0]]
    errors[1]=[i[0] for i in errors[1]]
    return(gpr2preds[0],errors[0][0],errors[1][0])


def get_preds_data_noMP(test,train,solvent):    
    preds=[]
    preds.append(get_pred_noMP("model_data/" + solvent + "/ET_noMP.dump",train,test))
    preds.append(get_pred_noMP("model_data/" + solvent + "/SVM_noMP.dump",train,test))
    preds.append(get_pred_noMP("model_data/" + solvent + "/MLP_noMP.dump",train,test))
    gpr_preds,lower,upper=get_pred_GPR_noMP("model_data/" + solvent + "/GPR_noMP.dump",train,test)
    preds.append(gpr_preds)
    mean=np.mean(preds)
    median=np.median(preds)
    preds.append(upper)
    preds.append(lower)
    preds.append(mean)
    preds.append(median)
    preds=[["ET","SVM","ANN","GP","GP_Upper","GP_Lower","Consensus_Mean","Consensus_Median"],["ExtraTrees Prediction","Support Vector Machine Prediction","Artificial Neural Networks Prediction","Gaussian Process Regression Prediction","Upper limit of GPR prediction (error to 1 SD)","Lower limit of GPR prediction (error to 1 SD)","Mean prediction from ET, SVM, ANN and GPR","Median prediction from ET, SVM, ANN and GPR"],preds]
    preds=np.array(preds).T.tolist()
    preds=pd.DataFrame(data=preds,columns=["Method","Algorithm Explanation","Prediction (LogS)"])
    return(preds)

##get xyz from sol log
code = {"1" : "H", "2" : "He", "3" : "Li", "4" : "Be", "5" : "B", \
"6"  : "C", "7"  : "N", "8"  : "O", "9" : "F", "10" : "Ne", \
"11" : "Na" , "12" : "Mg" , "13" : "Al" , "14" : "Si" , "15" : "P", \
"16" : "S"  , "17" : "Cl" , "18" : "Ar" , "19" : "K"  , "20" : "Ca", \
"21" : "Sc" , "22" : "Ti" , "23" : "V"  , "24" : "Cr" , "25" : "Mn", \
"26" : "Fe" , "27" : "Co" , "28" : "Ni" , "29" : "Cu" , "30" : "Zn", \
"31" : "Ga" , "32" : "Ge" , "33" : "As" , "34" : "Se" , "35" : "Br", \
"36" : "Kr" , "37" : "Rb" , "38" : "Sr" , "39" : "Y"  , "40" : "Zr", \
"41" : "Nb" , "42" : "Mo" , "43" : "Tc" , "44" : "Ru" , "45" : "Rh", \
"46" : "Pd" , "47" : "Ag" , "48" : "Cd" , "49" : "In" , "50" : "Sn", \
"51" : "Sb" , "52" : "Te" , "53" : "I"  , "54" : "Xe" , "55" : "Cs", \
"56" : "Ba" , "57" : "La" , "58" : "Ce" , "59" : "Pr" , "60" : "Nd", \
"61" : "Pm" , "62" : "Sm" , "63" : "Eu" , "64" : "Gd" , "65" : "Tb", \
"66" : "Dy" , "67" : "Ho" , "68" : "Er" , "69" : "Tm" , "70" : "Yb", \
"71" : "Lu" , "72" : "Hf" , "73" : "Ta" , "74" : "W"  , "75" : "Re", \
"76" : "Os" , "77" : "Ir" , "78" : "Pt" , "79" : "Au" , "80" : "Hg", \
"81" : "Tl" , "82" : "Pb" , "83" : "Bi" , "84" : "Po" , "85" : "At", \
"86" : "Rn" , "87" : "Fr" , "88" : "Ra" , "89" : "Ac" , "90" : "Th", \
"91" : "Pa" , "92" : "U"  , "93" : "Np" , "94" : "Pu" , "95" : "Am", \
"96" : "Cm" , "97" : "Bk" , "98" : "Cf" , "99" : "Es" ,"100" : "Fm", \
"101": "Md" ,"102" : "No" ,"103" : "Lr" ,"104" : "Rf" ,"105" : "Db", \
"106": "Sg" ,"107" : "Bh" ,"108" : "Hs" ,"109" : "Mt" ,"110" : "Ds", \
"111": "Rg" ,"112" : "Uub","113" : "Uut","114" : "Uuq","115" : "Uup", \
"116": "Uuh","117" : "Uus","118" : "Uuo"}

def get_xyz(file,save_file):
    f = open(file,"r")
    data = []
    for line in f:
        data.append(line)
    start=re.compile(r'Standard orientation')
    dash=re.compile(r'------------')
    lineno=[]
    lineno2=[]
    for f in range(len(data)):
        if re.findall(start,str(data[f])) != []:
            lineno.append(f)
    for f in range(len(data)):
        if re.findall(dash,str(data[f])) != []:
            lineno2.append(f)
    coord=[]
    linestart=lineno[-1]
    x=[]
    for f in range(len(lineno2)):
        if lineno2[f]>linestart:
            x.append(lineno2[f])
    lineend=x[2]
    for f in range(linestart,lineend):
        coord.append(data[f])
    for f in range(5):
        del(coord[0])
    for f in range(len(coord)):
        coord[f]=coord[f].split()
    #write xyz file
    f=open(save_file,'a')
    f.write(str(len(coord)) + '\n\n')
    for i in range(len(coord)):
        f.write(code[coord[i][1]] + ' ' + coord[i][3] + ' ' + coord[i][4] + ' ' + coord[i][5])
        if i==len(coord)-1:
            break
        f.write('\n')
    f.close()

# get the descriptors for the new molecule
def get_descs(path,gas_log,sol_log,solvent,MP,MW):
    if solvent == "water":
        L1=0.02492
        H1=-0.31927
    if solvent == "ethanol":
        L1=0.00021
        H1=-0.27815
    if solvent == "benzene":
        L1=-0.15891
        H1=-0.18032
    if solvent == "acetone":
        L1=-0.02797
        H1=-0.25801
    #gas log
    f = open(path + "/" + gas_log,"r") # open gas log
    data3 = []
    for line in f:
        data3.append(line)
    #G
    therm=re.compile(r'mal Free.*')
    for f in range(len(data3)):
        if re.findall(therm,str(data3[f])) != []:
            G_gas = re.findall(therm,str(data3[f]))
    therm2=re.compile(r'-?\d+\.\d+')
    G_gas=re.findall(therm2,str(G_gas))
    G_gas=str(G_gas)
    G_gas=G_gas.replace("['","")
    G_gas=G_gas.replace("']","")
    G_gas=float(G_gas)

    #HOMO
    HOMO=re.compile(r'Alpha  occ.*')
    for f in range(len(data3)):
        if re.findall(HOMO,str(data3[f])) != []:
            HO = re.findall(HOMO,str(data3[f]))
            j=f
    HOMO2=re.compile(r'-?\d+\.\d+')
    HO=re.findall(HOMO2,str(HO))[-1]
    HO=str(HO)
    HO=HO.replace("['","")
    HO=HO.replace("']","")
    HO=float(HO)
            
    #LUMO
    LUMO=re.compile(r'Alpha virt.*')
    LU = re.findall(LUMO,str(data3[j+1]))
    LUMO2=re.compile(r'-?\d+\.\d+')
    LU=re.findall(LUMO2,str(LU))[0]
    LU=str(LU)
    LU=LU.replace("['","")
    LU=LU.replace("']","")
    LU=float(LU)
    #LsolvHsolu and LsoluHsolv
    LsolvHsolu=L1-HO
    LsoluHsolv=LU-H1
    #sol opt file
    f = open(path + "/" + sol_log,"r") # open solv log
    data4 = []
    for line in f:
        data4.append(line)
    #dip
    dip=re.compile(r'Tot=.*')
    dip2=re.compile(r'-?\d+\.\d+')
    for f in range(len(data4)):
        if re.findall(dip,str(data4[f])) != []:
            solv_dipole = re.findall(dip,str(data4[f]))
    solv_dipole=re.findall(dip2,str(solv_dipole))
    solv_dipole=str(solv_dipole)
    solv_dipole=solv_dipole.replace("['","")
    solv_dipole=solv_dipole.replace("']","")
    solv_dipole=float(solv_dipole)
    #G
    for f in range(len(data4)):
        if re.findall(therm,str(data4[f])) != []:
            G_solv = re.findall(therm,str(data4[f]))
    G_solv=re.findall(therm2,str(G_solv))
    G_solv=str(G_solv)
    G_solv=G_solv.replace("['","")
    G_solv=G_solv.replace("']","")
    G_solv=float(G_solv)
    DeltaG_sol=G_solv-G_gas
                
    #SASA
    pymol.cmd.reinitialize()
    pymol.cmd.set('dot_solvent', 1)
    pymol.cmd.set('dot_density', 4)
    pymol.cmd.set('solvent_radius', 1.4)
    file_name=sol_log[:-4]
    pymol.cmd.load (path + "/" + file_name + ".xyz")
    ligand_area=pymol.cmd.get_area(file_name)
    pymol.cmd.delete(file_name)
    #volume
    vol=re.compile(r'Molar volume.*')
    for f in range(len(data4)):
        if re.findall(vol,str(data4[f])) != []:
            Volume = re.findall(vol,str(data4[f]))
    vol2=re.compile(r'\d+\.\d+')
    Volume=re.findall(vol2,str(Volume))[-1]
    Volume=str(Volume)
    Volume=Volume.replace("['","")
    Volume=Volume.replace("']","")
    Volume=float(Volume)
    #charges
    lineno=[]
    lineno2=[]
    summ=re.compile(r'Summary o') 
    equals=re.compile(r'\* Total \*')
    Mull=re.compile(r'Mulliken charges:')
    Mull2=re.compile(r'Sum of Mulliken charges')
    z=0
    for f in range(len(data4)):
        if re.findall(summ,str(data4[f])) !=[]:
            z=5
            for f in range(len(data4)):
                if re.findall(summ,str(data4[f])) != []:
                    lineno.append(f)
            for f in range(len(data4)):
                if re.findall(equals,str(data4[f])) != []:
                    lineno2.append(f)
            charge=[]
            for f in range(lineno[-1],lineno2[-1]+1):
                charge.append(data4[f])
            for f in range(len(charge)):
                charge[f]=charge[f].replace('\n','')
            #print charge
            for f in range(6):
                del(charge[0])
            del(charge[-1])
            del(charge[-1])
            for f in range(len(charge)):
                charge[f]=str.split(charge[f])
            for i in range(len(charge)):
                charge[i][2]=float(charge[i][2])
            #sum of charges on O
            O_charges = 0
            for i in range(len(charge)):
                if charge[i][0] == "O":
                    O_charges=O_charges+charge[i][2]
            #sum of charges on C
            C_charges = 0
            for i in range(len(charge)):
                if charge[i][0] == "C":
                    C_charges=C_charges+charge[i][2]
            #most negative atom
            Ordered = []
            for i in range(len(charge)):
                Ordered.append(charge[i][2])
            Ordered=sorted(Ordered)
            try:
                Most_negative = Ordered[0]
            except IndexError:
                Most_negative = "NA"
            #most positive atom
            try:
                Most_positive = Ordered[-1]
            except IndexError:
                Most_negative = "NA"
                                                        
            #sum of charges on non C/H
            Het_charges = 0
            for i in range(len(charge)):
                if charge[i][0] != "C":
                    if charge[i][0] != "H":
                        Het_charges=Het_charges+charge[i][2]
            break
    if z==0:
        for f in range(len(data4)):
            if re.findall(Mull,str(data4[f])) != []:
                lineno.append(f)
        for f in range(len(data4)):
            if re.findall(Mull2,str(data4[f])) != []:
                lineno2.append(f)
        charge=[]
        for f in range(lineno[-1],lineno2[-1]+1):
            charge.append(data4[f])
        for f in range(len(charge)):
            charge[f]=charge[f].replace('\n','')
        for f in range(2):
            del(charge[0])
        del(charge[-1])
        for f in range(len(charge)):
            charge[f]=str.split(charge[f])
        for i in range(len(charge)):
            charge[i][2]=float(charge[i][2])
        #sum of charges on O
        O_charges = 0
        for i in range(len(charge)):
            if charge[i][1] == "O":
                O_charges=O_charges+charge[i][2]
        #sum of charges on C
        C_charges = 0
        for i in range(len(charge)):
            if charge[i][1] == "C":
                C_charges=C_charges+charge[i][2]
        #most negative atom
        Ordered = []
        for i in range(len(charge)):
            Ordered.append(charge[i][2])
        Ordered=sorted(Ordered)
        try:
            Most_negative = Ordered[0]
        except IndexError:
            Most_negative = "NA"
        #most positive atom
        try:
            Most_positive = Ordered[-1]
        except IndexError:
            Most_negative = "NA"

        #sum of charges on non C/H
        Het_charges = 0
        for i in range(len(charge)):
            if charge[i][1] != "C":
                if charge[i][1] != "H":
                    Het_charges=Het_charges+charge[i][2]
    newcsvrow=[]
    newcsvrow.append(MW)
    newcsvrow.append(MP)
    newcsvrow.append(Volume)
    newcsvrow.append(G_solv)
    newcsvrow.append(DeltaG_sol)
    newcsvrow.append(solv_dipole)
    newcsvrow.append(LsoluHsolv)
    newcsvrow.append(LsolvHsolu)
    newcsvrow.append(ligand_area)
    newcsvrow.append(O_charges)
    newcsvrow.append(C_charges)
    newcsvrow.append(Most_negative)
    newcsvrow.append(Most_positive)
    newcsvrow.append(Het_charges)
    newcsvrow=pd.DataFrame(data=[newcsvrow],columns=['MW','MP','Volume','G_solv','DeltaG_sol','solv_dip',
                     'LsoluHsolv','LsolvHsolu','SASA','O_charges',
                     'C_charges','Most_neg','Most_pos','Het_charges'])
    return(newcsvrow)


# get MW from xyz
def get_MW(file):
    mol=Molecule.from_file(file)
    mol.set_default_masses()
    return(mol.mass/amu)

#detect/verify the solvent
def get_solvent(file):
    reg1='SCRF=\(solvent=.*\)'
    f = open(file,"r") # open sol file
    data = []
    for line in f:
        data.append(line)
    for f in range(len(data)):
        if re.findall(reg1,str(data[f])) != []:
            solvent = re.findall(reg1,str(data[f]))
    solvent=solvent[0]
    solvent=solvent.replace("SCRF=(solvent=","")
    solvent=solvent.replace(")","")
    return(solvent.lower())

#get the final prediction
def get_final_pred(path,gas_log,sol_log,MP):
    # get the xyz
    get_xyz(path + "/" + sol_log,path + "/" + sol_log.replace(".log",".xyz"))
    # get the MW
    MW=get_MW(path + "/" + sol_log.replace(".log",".xyz"))
    # check the solvent
    solvent=get_solvent(path + "/" + sol_log)
    # get the descriptors for the new molecule
    new_mol=get_descs(path,gas_log,sol_log,solvent,MP,MW)
    # get right solvent and whether MP is included and return data
    if solvent == "water":
        if MP != "":
            return("Water",get_preds_data(new_mol,water_train,solvent))
        else:
            return("Water",get_preds_data_noMP(new_mol,water_train,solvent))
    if solvent == "ethanol":
        if MP != "":
            return("Ethanol",get_preds_data(new_mol,ethanol_train,solvent))
        else:
            return("Ethanol",get_preds_data_noMP(new_mol,ethanol_train,solvent))
    if solvent == "benzene":
        if MP != "":
            return("Benzene",get_preds_data(new_mol,benzene_train,solvent))
        else:
            return("Benzene",get_preds_data_noMP(new_mol,benzene_train,solvent))
    if solvent == "acetone":
        if MP != "":
            return("Acetone",get_preds_data(new_mol,acetone_train,solvent))
        else:
            return("Acetone",get_preds_data_noMP(new_mol,acetone_train,solvent))
