%nprocshared=4
%mem=100MW
%chk=WXZMFSXDPGVJKK-UHFFFAOYSA-N.chk
#b3lyp/6-31+G(d) opt freq

gas opt

0 1
O    1.0794    2.1615   -0.0583
C    0.5581    1.1176   -0.8833
C    0.0000    0.0000    0.0000
C   -0.5581   -1.1176   -0.8833
O   -1.0794   -2.1615   -0.0583
C    1.1176   -0.5581    0.8833
O    2.1615   -1.0794    0.0583
C   -1.1176    0.5581    0.8833
O   -2.1615    1.0794    0.0583
H    1.4507    2.9051   -0.5525
H   -0.2381    1.5152   -1.5127
H    1.3543    0.7200   -1.5127
H   -1.3543   -0.7200   -1.5127
H    0.2381   -1.5152   -1.5127
H   -1.4507   -2.9051   -0.5525
H    1.5152    0.2381    1.5127
H    0.7200   -1.3543    1.5127
H    2.9051   -1.4507    0.5525
H   -1.5152   -0.2381    1.5127
H   -0.7200    1.3543    1.5127
H   -2.9051    1.4507    0.5525


