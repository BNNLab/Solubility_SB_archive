%nprocshared=4
%mem=100MW
%chk=YCLSOMLVSHPPFV-UHFFFAOYSA-N.chk
#b3lyp/6-31+G(d) opt freq

gas opt

0 1
S   -0.3829   -1.2411    0.9508
S    0.3829   -1.2411   -0.9508
C    1.6430    0.0621   -0.8832
C    2.8143   -0.4055   -0.0170
C    3.8611    0.6771    0.0392
O    3.6881    1.7161   -0.5529
O    4.9857    0.4872    0.7470
C   -1.6430    0.0621    0.8832
C   -2.8143   -0.4055    0.0170
C   -3.8611    0.6771   -0.0392
O   -3.6881    1.7161    0.5529
O   -4.9857    0.4872   -0.7470
H    1.9991    0.2758   -1.8910
H    1.2097    0.9645   -0.4520
H    2.4582   -0.6192    0.9908
H    3.2476   -1.3079   -0.4482
H    5.6279    1.2102    0.7532
H   -1.9991    0.2758    1.8910
H   -1.2097    0.9645    0.4520
H   -2.4582   -0.6192   -0.9908
H   -3.2476   -1.3079    0.4482
H   -5.6279    1.2102   -0.7532


