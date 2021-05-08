from rdkit import Chem
from rdkit.Chem import AllChem
import rdkit

# getting input file method
#gas commands
commands_gas="#b3lyp/6-31+G(d) opt freq"
#sol commands
commands_sol="#b3lyp/6-31+G(d) opt freq volume pop=nbo SCRF=(solvent="
#get gas file
def get_input_file_gas(smi,ESM):
    # modify if PM6
    if ESM == "PM6":
        commands=commands_gas.replace("b3lyp/6-31+G(d)","PM6")
        commands=commands.replace("pop=nbo ","")
    else:
        commands=commands_gas
    # molecular object
    m=Chem.MolFromSmiles(smi)
    # try to build 3d structure or return error
    try:
        m=Chem.AddHs(m)
        AllChem.EmbedMolecule(m)
        m=rdkit.Chem.rdmolfiles.MolToXYZBlock(m)
    except:
        return(["Error in SMILES"])
    try:
        m=m.splitlines()
    except AttributeError:
        return(["Error in SMILES"])
    # build input file
    del(m[0])
    del(m[0])
    pos=smi.count('+')
    neg=smi.count('-')
    charge=(-neg)+pos
    f = []
    f.append("%nprocshared=4")
    f.append("%mem=100MW")
    f.append("%NoSave")
    f.append("%chk=gas.chk")
    f.append(commands)
    f.append("")
    f.append("gas opt")
    f.append("")
    f.append(str(charge) + " 1")
    for i in range(len(m)):
        f.append(m[i])
    f.append("")
    return(f)##returns a list of every line in file
#get sol file
def get_input_file_sol(smi,solvent,ESM):
    if ESM == "PM6":
        commands=commands_sol.replace("b3lyp/6-31+G(d)","PM6")
        commands=commands.replace("pop=nbo ","")
    else:
        commands=commands_sol
    m=Chem.MolFromSmiles(smi)
    try:
        m=Chem.AddHs(m)
        AllChem.EmbedMolecule(m)
        m=rdkit.Chem.rdmolfiles.MolToXYZBlock(m)
    except:
        return(["Error in SMILES"])
    try:
        m=m.splitlines()
    except AttributeError:
        return(["Error in SMILES"])
    del(m[0])
    del(m[0])
    pos=smi.count('+')
    neg=smi.count('-')
    charge=(-neg)+pos
    f = []
    f.append("%nprocshared=4")
    f.append("%mem=100MW")
    f.append("%NoSave")
    f.append("%chk=sol.chk")
    f.append(commands + str(solvent) + ")")
    f.append("")
    f.append("sol opt")
    f.append("")
    f.append(str(charge) + " 1")
    for i in range(len(m)):
        f.append(m[i])
    f.append("")
    return(f)##returns a list of every line in file