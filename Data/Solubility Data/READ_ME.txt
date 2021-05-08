COLUMN KEY FOR SOLUBILITY DATA EXCLUDING WATER LARGE
===============================================================================
StdInChIKey = StdInChIKey for molecule; unique identifier
SMILES = SMILES code for molecule
LogS = Solubility (log base 10 C, where C = concentration in units mol/L)
T = Temperature of measurement
Source = Source of solubility measurement
REF1, REF2, REF3 etc. = Reference(s) for solubility measurement; only for water

COLUMN KEY FOR SOLUBILITY DATA FOR WATER LARGE
===============================================================================
StdInChIKey = StdInChIKey for molecule; unique identifier
SMILES = SMILES code for molecule
Dataset1 = Dataset of first solubility value
LogS1 = First solubility value (log base 10 C, where C = concentration in units mol/L)
T1 = Temperature of first solubility measurement
Dataset2 = Dataset of second solubility value
LogS2 = Second solubility value (log base 10 C, where C = concentration in units mol/L)
T2 = Temperature of second solubility measurement
Dataset1 = Dataset of third solubility value
LogS1 = Third solubility value (log base 10 C, where C = concentration in units mol/L)
T1 = Temperature of third solubility measurement
N_datapoints = Number of data point (1, 2, or 3)
SD_error = Standard deviation of solubility (if 3 values)
Error_note = Note on the standard deviation and if this led to a value being removed
Pred_comment = Comment on the initial 10-fold cross validation prediction
LogS Changed? = Whether the initial 10-fold cross validation led to a change in LogS used
LogS_mean = Mean value of solubility
LogS_median = Median value of solubility