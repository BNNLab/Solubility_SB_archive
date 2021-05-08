'''
This python script gets the charged surface descriptors from the data gathered from following ChargedSurfaceDescriptors.txt. This is done by clustering areas of high charge and generating statistics about them.
INPUTS:
ESP_folder - directory with the previously gathered ESP data
ESP_clusters - directory to save data about the charges regions
ESP_descs.csv - location and name of file to save descriptors
OUTPUTS:
ESP_descs.csv - a .csv file containing descriptors
'''
##section 1: import modules
import os,re
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hcluster

##section 2: define inputs and outputs
ESP_folder="wfn_esp" #input ESP data directory
ESP_clusters="esp_cluster" #output directory for clusters
os.makedirs(ESP_clusters)
ESP_descs="espdesc.csv" #output file for descriptors

##section 3: gets clusters and then descriptors for each molecule
#step 1: produce .csv file detailing area, mean charge and charge standard deviation for each region
#do for all molecules
for files in os.listdir(ESP_folder):
    data=pd.read_csv(ESP_folder + "\\" + files)
    files=files.replace(".csv","")
    x=list(data['X'])
    y=list(data['Y'])
    z=list(data['Z'])
    ESP=list(data['ESP'])
    #filter "neutral points"
    #limits
    lim1=-0.02
    lim2=0.02
    #new lists
    ESP1=[];ESP2=[];x1=[];x2=[];y1=[];y2=[];z1=[];z2=[]
    #ESP1 etc. are the "keep" lists
    for f in range(len(ESP)):
        if ESP[f] > lim2: #if within limits
            ESP1.append(ESP[f])
            x1.append(x[f])
            y1.append(y[f])
            z1.append(z[f])
        elif ESP[f] < lim1:
            ESP1.append(ESP[f])
            x1.append(x[f])
            y1.append(y[f])
            z1.append(z[f])        
    #ESP2 etc. are the "discard" lists
    for f in range(len(ESP)):
        if ESP[f] < lim2:
            if ESP[f] > lim1:
                ESP2.append(ESP[f])
                x2.append(x[f])
                y2.append(y[f])
                z2.append(z[f])
    ###if no point left
    if len(x1) == 0:
        charge_data=[[0,0,0,0]] #give zeros for stats
        df_charge=pd.DataFrame(data=charge_data,columns=["Number","Area","Mean_charge","Std_charge"])
        df_charge.to_csv(dir2 + "\\" + files + ".csv") #save file
        continue
    #get list of coordinates form list of x, y and z
    coord=[]
    for f in range(len(x1)):
        xyz=[]
        xyz.append(x1[f])
        xyz.append(y1[f])
        xyz.append(z1[f])
        coord.append(xyz)
    #define a threshold for finding regions
    thresh = 0.8
    clusters = hcluster.fclusterdata(coord, thresh, criterion="distance")
    #clusters are now labels
    clusters=[str(x) for x in clusters]
    #add cluster and charge to data
    for f in range(len(coord)):
        coord[f].append(clusters[f])
        coord[f].append(ESP1[f])
    df=pd.DataFrame(data=coord,columns=['x','y','z','cluster','ESP'])
    #group data by cluster
    groups = df.groupby('cluster')
    #get charge areas, mean and std
    charge_data=[]
    for name, group in groups:
        charge=group.ESP
        charge_mean=np.mean(charge)
        charge_std=np.std(charge)
        area=len(charge)
        data=[] #add to new list
        data.append(name)
        data.append(area)
        data.append(charge_mean)
        data.append(charge_std)
        charge_data.append(data)
    #save to file
    df_charge=pd.DataFrame(data=charge_data,columns=["Number","Area","Mean_charge","Std_charge"])
    df_charge = df_charge.sort_values('Area')
    df_charge.to_csv(ESP_clusters + "\\" + files + ".csv")
#step 2: get descriptors
dir1="esp_cluster" #.csv files of cluster stats
data_set=[]
for files in os.listdir(dir1): #for each molecule
    df=pd.read_csv(dir1 + "\\" + files)
    data=[]
    #append StdInChIKey
    files=files.replace('.csv','')
    data.append(files)
    #get number of points per region
    no_regions=df["Area"]
    no_regions_trimmed=[]
    #append no. of regions with more than 20 points
    for f in no_regions:
        if f > 20:
            no_regions_trimmed.append(f)
    data.append(len(no_regions_trimmed))
    #append total charge weighted area
    points=df["Area"]
    charge=df["Mean_charge"]
    tot_weighted_charge=0
    for f in range(len(points)):
        weighted_charge=points[f]*charge[f]
        tot_weighted_charge=tot_weighted_charge+weighted_charge
    data.append(tot_weighted_charge)
    #append total neg charge
    neg_weighted_charge=0
    for f in range(len(points)):
        if charge[f]<0:
            weighted_charge=points[f]*charge[f]
            neg_weighted_charge=neg_weighted_charge+weighted_charge
    data.append(neg_weighted_charge)
    #append total pos charge
    pos_weighted_charge=0
    for f in range(len(points)):
        if charge[f]>0:
            weighted_charge=points[f]*charge[f]
            pos_weighted_charge=pos_weighted_charge+weighted_charge
    data.append(pos_weighted_charge)
    #append information about largest region
    big_index=np.argmax(np.array(points))
    data.append(points[big_index])
    data.append(charge[big_index])
    std=df['Std_charge']
    data.append(std[big_index])
    #append to master list
    data_set.append(data)
#create dataframe of descriptors
data_df=pd.DataFrame(data=data_set,columns=["StdInChIKey","No_regions","Tot_charge","Neg_charge","Pos_charge","Big_area","Big_charge","Big_std"])
data_df.to_csv(ESP_descs,index=False) #save file
