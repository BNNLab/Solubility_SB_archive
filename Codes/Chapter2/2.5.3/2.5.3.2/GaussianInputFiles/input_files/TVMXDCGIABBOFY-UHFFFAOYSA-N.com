%nprocshared=4
%mem=100MW
%chk=TVMXDCGIABBOFY-UHFFFAOYSA-N.chk
#b3lyp/6-31+G(d) opt freq

gas opt

0 1
C    0.6076   -0.4648   -0.0000
C   -0.6076    0.4648   -0.0000
C   -1.8891   -0.3710    0.0000
C   -3.1043    0.5587   -0.0000
C   -4.3858   -0.2771    0.0000
C    1.8891    0.3710    0.0000
C    3.1043   -0.5587   -0.0000
C    4.3858    0.2771    0.0000
H    0.5839   -1.0937    0.8900
H    0.5839   -1.0937   -0.8900
H   -0.5839    1.0937    0.8900
H   -0.5839    1.0937   -0.8900
H   -1.9128   -0.9998   -0.8900
H   -1.9128   -0.9998    0.8900
H   -3.0807    1.1875    0.8900
H   -3.0807    1.1875   -0.8900
H   -4.4095   -0.9060    0.8900
H   -5.2516    0.3852    0.0000
H   -4.4095   -0.9060   -0.8900
H    1.9128    0.9998   -0.8900
H    1.9128    0.9998    0.8900
H    3.0807   -1.1875    0.8900
H    3.0807   -1.1875   -0.8900
H    5.2516   -0.3852   -0.0000
H    4.4095    0.9060   -0.8900
H    4.4095    0.9060    0.8900


