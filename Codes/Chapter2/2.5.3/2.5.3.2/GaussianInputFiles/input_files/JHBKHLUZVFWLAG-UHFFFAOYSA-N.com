%nprocshared=4
%mem=100MW
%chk=JHBKHLUZVFWLAG-UHFFFAOYSA-N.chk
#b3lyp/6-31+G(d) opt freq

gas opt

0 1
Cl   -2.7013   -1.5603   -0.0009
C   -1.1978   -0.6923   -0.0008
C   -1.1978    0.6923   -0.0003
Cl   -2.7013    1.5603    0.0003
C    0.0000    1.3839    0.0002
C    1.1979    0.6923   -0.0003
Cl    2.7013    1.5603   -0.0002
C    1.1979   -0.6923   -0.0003
Cl    2.7013   -1.5603   -0.0004
C   -0.0000   -1.3839    0.0045
H    0.0000    2.4639    0.0006
H   -0.0000   -2.4639    0.0045

