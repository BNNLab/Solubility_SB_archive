#$ -cwd -V
#$ -l h_vmem=1G
#$ -l h_rt=24:00:00
#$ -l disk=4G
#$ -pe smp 4
#$ -m be
module load gaussian
export GAUSS_SCRDIR=$TMPDIR
g09 FHADSMKORVFYOS-UHFFFAOYSA-N.com
rm ${GAUSS_SCRDIR}/*