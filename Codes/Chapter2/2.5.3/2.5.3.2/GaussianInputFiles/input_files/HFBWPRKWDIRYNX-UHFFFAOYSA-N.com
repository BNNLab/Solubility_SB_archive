%nprocshared=4
%mem=100MW
%chk=HFBWPRKWDIRYNX-UHFFFAOYSA-N.chk
#b3lyp/6-31+G(d) opt freq

gas opt

0 1
Cl   -0.7737   -3.2627    0.3113
C   -0.4961   -1.5569    0.1467
N    0.7412   -1.0885    0.1249
C    0.9572    0.2183   -0.0016
N    2.2479    0.7060   -0.0247
C    3.3812   -0.2152    0.0902
C    3.8044   -0.6797   -1.3048
C    2.4849    2.1450   -0.1640
C    2.5387    2.7896    1.2225
N   -0.0712    1.0533   -0.1063
C   -1.3124    0.5799   -0.0841
N   -2.3831    1.4436   -0.1929
C   -3.7514    0.9208   -0.1684
C   -4.7423    2.0785   -0.3051
N   -1.5219   -0.7288    0.0377
H    4.2158    0.2938    0.5725
H    3.0884   -1.0788    0.6874
H    2.9699   -1.1887   -1.7870
H    4.0973    0.1839   -1.9019
H    4.6476   -1.3651   -1.2193
H    3.4320    2.3091   -0.6780
H    1.6758    2.5922   -0.7415
H    1.5916    2.6255    1.7365
H    3.3478    2.3424    1.8000
H    2.7150    3.8603    1.1188
H   -2.2298    2.3969   -0.2856
H   -3.8894    0.2248   -0.9959
H   -3.9257    0.4029    0.7748
H   -4.6044    2.7744    0.5224
H   -4.5681    2.5964   -1.2483
H   -5.7604    1.6895   -0.2869


