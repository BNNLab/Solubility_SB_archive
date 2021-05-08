'''
Python script to extract descriptors from Gaussain 09 ouput files.
This script extracts: every descriptor except MW (see separate script) and MP (experimental).
Data must be organised into two folders: "gas/" with the gas .log files and "sol/" with the solution .log files within.
The names of the files become the names of the molecules and must be the same for both gas and solution .log file.
INPUTS:
OutputFiles/ - directory with gas/ and sol/ directories within, containing the output .log files from Gaussian 09
OUTPUTS:
Descriptors.csv - New .csv file containing the descriptors calculated
'''
##Section 1: import modules
import __main__
__main__.pymol_argv = ['pymol','-qc'] # Pymol: quiet and no GUI
import pymol,sys,os,re
pymol.finish_launching()
import sys,os,re,pickle
import pandas as pd
import numpy as np

##section 2: define folder of output files and name of new descriptor file
dir=os.path.dirname(__file__)#get current directory to join to files
Directory=os.path.join(dir,"OutputFiles")
Output=os.path.join(dir,"Descriptors.csv")

##section 3: define methods
##get xyz from sol log
##convert atomic number to symbol using this dictionary
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


##determine the solvent used in the solution optimisation
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

##get the descriptors using regular expressions
def get_descs(gas_log,sol_log,solvent,files):
	##HOMO and LUMO for solvents (previosuly calculated for the same method)
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
	#open gas log
	f = open(gas_log,"r") # open gas log
	data3 = []
	for line in f:
		data3.append(line)
	#get E0   
	scfd=re.compile(r'SCF D.*')
	for f in range(len(data3)):
		if re.findall(scfd,str(data3[f])) != []:
			E0_gas = re.findall(scfd,str(data3[f]))
	scfd2=re.compile(r'-?\d+\.\d+E?-?\d?\d?')
	E0_gas=re.findall(scfd2,str(E0_gas))
	E0_gas=str(E0_gas)
	E0_gas=E0_gas.replace("['","")
	E0_gas=E0_gas.replace("']","")
	E0_gas=float(E0_gas)
	#get dipole
	dip=re.compile(r'Tot=.*')
	for f in range(len(data3)):
		if re.findall(dip,str(data3[f])) != []:
			gas_dipole = re.findall(dip,str(data3[f]))
	dip2=re.compile(r'-?\d+\.\d+')
	gas_dipole=re.findall(dip2,str(gas_dipole))
	gas_dipole=str(gas_dipole)
	gas_dipole=gas_dipole.replace("['","")
	gas_dipole=gas_dipole.replace("']","")
	gas_dipole=float(gas_dipole)
	#get G
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
	#get HOMO
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
	#get LUMO
	LUMO=re.compile(r'Alpha virt.*')
	LU = re.findall(LUMO,str(data3[j+1]))
	LUMO2=re.compile(r'-?\d+\.\d+')
	LU=re.findall(LUMO2,str(LU))[0]
	LU=str(LU)
	LU=LU.replace("['","")
	LU=LU.replace("']","")
	LU=float(LU)
	#get LsolvHsolu and LsoluHsolv
	LsolvHsolu=L1-HO
	LsoluHsolv=LU-H1
	#open solution log file
	f = open(sol_log,"r") # open solv log
	data4 = []
	for line in f:
		data4.append(line)
	#get E0 and DeltaE0sol
	for f in range(len(data4)):
		if re.findall(scfd,str(data4[f])) != []:
			E0_solv = re.findall(scfd,str(data4[f]))
	E0_solv=re.findall(scfd2,str(E0_solv))
	E0_solv=str(E0_solv)
	E0_solv=E0_solv.replace("['","")
	E0_solv=E0_solv.replace("']","")
	E0_solv=float(E0_solv)
	DeltaE0_sol=E0_solv-E0_gas
	#get dipole
	for f in range(len(data4)):
		if re.findall(dip,str(data4[f])) != []:
			solv_dipole = re.findall(dip,str(data4[f]))
	solv_dipole=re.findall(dip2,str(solv_dipole))
	solv_dipole=str(solv_dipole)
	solv_dipole=solv_dipole.replace("['","")
	solv_dipole=solv_dipole.replace("']","")
	solv_dipole=float(solv_dipole)
	#get G and DeltaGsol
	for f in range(len(data4)):
		if re.findall(therm,str(data4[f])) != []:
			G_solv = re.findall(therm,str(data4[f]))
	G_solv=re.findall(therm2,str(G_solv))
	G_solv=str(G_solv)
	G_solv=G_solv.replace("['","")
	G_solv=G_solv.replace("']","")
	G_solv=float(G_solv)
	DeltaG_sol=G_solv-G_gas
	#find the no. of atoms
	NAtom1=re.compile(r'NAtoms.*')
	for f in range(len(data4)):
		if re.findall(NAtom1,str(data4[f])) != []:
			NAt = re.findall(NAtom1,str(data4[f]))
	NAtom2=re.compile(r'\d+')
	NAt=re.findall(NAtom2,str(NAt))[0]
	NAt=str(NAt)
	NAt=NAt.replace("['","")
	NAt=NAt.replace("']","")
	NAt=int(NAt)
	#get SASA using pymol
	pymol.cmd.reinitialize()
	pymol.cmd.set('dot_solvent', 1)
	pymol.cmd.set('dot_density', 4)
	pymol.cmd.set('solvent_radius', 1.4)##this is the solvent radius (1.4 ang)
	file_name=files[:-4]
	pymol.cmd.load (sol_log.replace(".log",".xyz"))
	ligand_area=pymol.cmd.get_area(file_name)
	pymol.cmd.delete(file_name)
	#get volume
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
	#get charge descriptors (npa if specified, otherwise Mulliken)
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
	##assemble new row and return it
	newcsvrow=[]
	newcsvrow.append(files.replace(".log",""))
	newcsvrow.append(E0_gas)
	newcsvrow.append(E0_solv)
	newcsvrow.append(DeltaE0_sol)
	newcsvrow.append(G_gas)
	newcsvrow.append(G_solv)
	newcsvrow.append(DeltaG_sol)
	newcsvrow.append(HO)
	newcsvrow.append(LU)
	newcsvrow.append(LsoluHsolv)
	newcsvrow.append(LsolvHsolu)
	newcsvrow.append(gas_dipole)
	newcsvrow.append(solv_dipole)
	newcsvrow.append(O_charges)
	newcsvrow.append(C_charges)
	newcsvrow.append(Most_negative)
	newcsvrow.append(Most_positive)
	newcsvrow.append(Het_charges)
	newcsvrow.append(Volume)
	newcsvrow.append(ligand_area)
	newcsvrow.append(NAt)
	return(newcsvrow)

##getting the final set of descriptors
def get_final_descs(directory,Output):
	#master list
	master=[]
	##for every file in gas directory
	for files in os.listdir(os.path.join(directory,"gas")):
		##get xys for the sol .log (needed for SASA)
		get_xyz(os.path.join(directory,"sol",files),os.path.join(directory,"sol",files.replace(".log",".xyz")))
		##find solvent used
		solvent=get_solvent(os.path.join(directory,"sol",files))
		##get descriptors and add to master list
		new_mol=get_descs(os.path.join(directory,"gas",files),os.path.join(directory,"sol",files),solvent,files)
		master.append(new_mol)
	##create final dataframe and save
	descs=pd.DataFrame(data=master,columns=['Name','E0_gas','E0_solv','DeltaE0_sol','G_gas','G_solv',
	'DeltaG_sol','HOMO','LUMO','LsoluHsolv','LsolvHsolu','Gas_dip','Solv_dip','O_charges',
	'C_charges','Most_neg','Most_pos','Het_charges','Volume','SASA','N_atoms'])
	descs.to_csv(Output,index=False)

##section 4: run final method to get descriptors and save
get_final_descs(Directory,Output)