##This R script calculates Q2 (predictivity) of successive principal components for a dataset
##INPUTS:
##INPUTS:
##Descriptors.csv - a .csv file containing calculated descriptors for a dataset, the descriptors must be named as in the global variable "Descs"
##OUTPUTS:
##Descriptors_Q2.csv - a .csv containing the Q2 values for successive principal components, first column is the principal component number and the second column is the Q2 value

##section 1: import libraries
library("pcaMethods")
library("funr")

##section 2: define inputs and outputs
dir <- funr::get_script_path()##get path to directory .r script is in
Descriptors <- "Descriptors.csv"##name of file containing descriptors
Descriptors_Q2 <- "Descriptors_Q2.csv"##name of output file

##section 3: define method for getting Q2
##column names of descriptors
descs <- c('E0_gas','E0_solv','DeltaE0_sol','G_gas','G_solv','DeltaG_sol','HOMO','LUMO','LsoluHsolv','LsolvHsolu',
       'gas_dip','solv_dip','O_charges','C_charges','Most_neg','Most_pos','Het_charges','Volume','SASA','MW','N_atoms','MP')
##define method
get_Q2 <- function(dir,Descriptors,Descriptors_Q2){
	##read in data
	data <- read.csv(file.path(dir,Descriptors))
	##get just descriptors
	x <- data[descs]
	##run PCA analysis
	pcIr <- pca(x, nPcs=22, cv = "q2", scale="vector")
	##get Q2 values
	q2 <- Q2(pcIr, x)
	##write output to file
	write.csv(q2,file.path(dir,Descriptors_Q2), row.names = TRUE)
}

##section 4: run method to get Q2
get_Q2(dir,Descriptors,Descriptors_Q2)