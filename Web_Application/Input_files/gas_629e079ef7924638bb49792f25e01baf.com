%nprocshared=4
%mem=100MW
%NoSave
%chk=gas.chk
#PM6 opt freq

gas opt

0 1
C      0.794262   -1.148974    0.015000
C      1.360699    0.129837   -0.002163
C      0.589827    1.268073   -0.017027
C     -0.786625    1.145009   -0.014949
C     -1.341315   -0.112392    0.001925
C     -0.580623   -1.265551    0.016991
H      1.402552   -2.042792    0.026673
H      2.418314    0.211316   -0.003586
H      1.007452    2.268593   -0.030447
H     -1.418248    2.029280   -0.026492
H     -2.422241   -0.217238    0.003669
H     -1.024054   -2.265159    0.030406

