%nprocshared=4
%mem=100MW
%chk=WLJVNTCWHIRURA-UHFFFAOYSA-N.chk
#b3lyp/6-31+G(d) opt freq

gas opt

0 1
O    3.6181   -1.3087    0.0010
C    3.7289   -0.1055    0.0016
O    4.9481    0.4561   -0.0034
C    2.4985    0.7645    0.0016
C    1.2492   -0.1188    0.0009
C    0.0000    0.7645    0.0010
C   -1.2492   -0.1188    0.0002
C   -2.4985    0.7645    0.0003
C   -3.7289   -0.1055   -0.0005
O   -3.6181   -1.3087   -0.0010
O   -4.9481    0.4561   -0.0005
H    5.7073   -0.1427   -0.0074
H    2.4982    1.3934    0.8919
H    2.4987    1.3942   -0.8881
H    1.2495   -0.7477   -0.8894
H    1.2490   -0.7485    0.8906
H   -0.0002    1.3934    0.8912
H    0.0002    1.3942   -0.8887
H   -1.2490   -0.7477   -0.8900
H   -1.2495   -0.7485    0.8899
H   -2.4987    1.3934    0.8905
H   -2.4982    1.3942   -0.8894
H   -5.7073   -0.1427   -0.0010

