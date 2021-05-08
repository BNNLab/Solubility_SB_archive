'''
This python script takes SMILES in an input code and generates the relevant input files for Gaussian 09 in the gas phase.
N.B. The final gas structure was transferred in situ and the following keywords were added to get input files for solution structures: "pop=nbo", "volume" and "SCRF=(solvent=sol)" where "sol" is the required solvent.
N.B. For iodine molecules, 6-31+G(d) was changed to lanl2dz but only for the iodine molecules. This was done by hand.
IMPORTANT: CIRPy is used to generate the initial 3D structures. Some 3D start structures were generated manually if CIRpy failed to come up with a plausible structure. In this case the name of the molecule is printed and the input files are not generated.
INPUTS:
dataset.csv - dataset with a column called "SMILES" with SMILES codes and a column called "Name" containing a unique identifier
OUTPUTS:
input_files/ - a new directory with input files called by the name of each molecule
'''
##section 1: import modules
import cirpy
import os
import pandas as pd

##section 2: define input, output files and Gaussian commands
dir=os.path.dirname(__file__)#get current directory to join to files
dataset=pd.read_csv(os.path.join(dir,"dataset.csv"))#location of dataset file
output_dir=os.path.join(dir,"input_files")#location of output directory for input files
##commands for Gaussian
commands="#b3lyp/6-31+G(d) opt freq"
##time limit for calculation (max 48h)
time="24:00:00"

##section 3: method for generating files
def InputFiles(data,output_file_dir,commands,time):
	os.mkdir(output_file_dir)
	##get SMILES list
	smi2=data["SMILES"]
	##get Name
	name2=data["Name"]
	##for each molecule
	for j in range(len(smi2)):
		##get name and smiles for this molecule
		name = name2[j]
		smi = smi2[j]
		##attempt to get initial 3D struture
		try:
			m = cirpy.resolve(smi,'xyz')
		##if it cannot, print name and move to next molecule
		except:
			print(name)
			continue
		##attempt to split xyz coordinates
		try:
			m=m.splitlines()
		##if it cannot, print name and move to next molecule
		except AttributeError:
			print(name)
			continue
		##delete first 2 lines of xyz (always contian title and number of atoms)
		del(m[0])
		del(m[0])
		##determine charge from SMILES code
		pos=smi.count('+')
		neg=smi.count('-')
		charge=(-neg)+pos
		##open .com input file and write contents line by line
		f = open(os.path.join(output_file_dir,name + ".com"),"a")
		f.write("%nprocshared=4\n")
		f.write("%mem=100MW\n")
		f.write("%chk=" + name + ".chk\n")
		f.write(commands + "\n")
		f.write("\ngas opt\n\n")
		f.write(str(charge) + " 1\n")
		##write every line of xyz
		for i in range(len(m)):
			f.write(m[i])
			f.write('\n')
		f.write('\n')
		f.write('\n')
		f.close()
		
		##open .sh file and write contents line by line
		f = open(os.path.join(output_file_dir,name + ".sh"),"a")
		f.write("#$ -cwd -V\n")
		f.write("#$ -l h_vmem=1G\n")
		f.write("#$ -l h_rt=" + time + "\n")
		f.write("#$ -l disk=4G\n")
		f.write("#$ -pe smp 4\n")
		f.write("#$ -m be\n")
		f.write("module load gaussian\n")
		f.write("export GAUSS_SCRDIR=$TMPDIR\n")
		f.write("g09 " + name + ".com\n")
		f.write("rm ${GAUSS_SCRDIR}/*")
		f.close()

##section 4: run method and get input files
InputFiles(dataset,output_dir,commands,time)