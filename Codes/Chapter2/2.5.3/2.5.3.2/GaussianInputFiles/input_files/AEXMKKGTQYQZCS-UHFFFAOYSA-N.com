%nprocshared=4
%mem=100MW
%chk=AEXMKKGTQYQZCS-UHFFFAOYSA-N.chk
#b3lyp/6-31+G(d) opt freq

gas opt

0 1
C    0.0000   -0.0000    0.0178
C   -1.2492    0.0000    0.9011
C   -2.4985   -0.0000    0.0178
C    1.2492   -0.0000    0.9011
C    2.4985    0.0000    0.0178
C   -0.0000   -1.2492   -0.8656
C    0.0000    1.2492   -0.8656
H   -1.2492   -0.8900    1.5304
H   -1.2492    0.8900    1.5304
H   -2.4985    0.8900   -0.6115
H   -2.4985   -0.8900   -0.6115
H   -3.3885    0.0000    0.6471
H    1.2492    0.8900    1.5304
H    1.2492   -0.8900    1.5304
H    2.4985   -0.8900   -0.6115
H    2.4985    0.8900   -0.6115
H    3.3885    0.0000    0.6471
H   -0.8900   -1.2492   -1.4949
H    0.8900   -1.2492   -1.4949
H   -0.0000   -2.1392   -0.2363
H    0.0000    2.1392   -0.2363
H    0.8900    1.2492   -1.4949
H   -0.8900    1.2492   -1.4949


