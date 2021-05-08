'''
This python script takes optimised cartesian coordinates for solution structures, generates shadow images from three perpendicular angles, and saves them as .png files.
INPUTS:
xyz_folder - directory with the xyz coordinates of optimised structures
vdw_points - directory to save the points generated on the VdW surface
image_folder - directory where image files are saved
OUTPUTS:
Three .png files for every molecule showing the shadow from three perpendicular angles
'''
##section 1: import modules
import math,re,os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

##section 2: define inputs and outputs
xyz_folder="xyz"#xyz files location
vdw_points="vdw_points"#where points on vdw will be saved
os.makedirs(vdw_points)
image_folder="shadow_image_vdw"#where images will be saved
os.makedirs(image_folder)
##number of points per atom
number_points=1000
##van der waal radii
radiiDict={'H':1.2,'B':2,'C':1.7,'N':1.55,'O':1.52,'F':1.47,'P':1.8,'S':1.8,'Cl':1.75,'Br':1.85,'I':1.98}

##section 3: define method to get points and save files
##get points on vdw surface
def get_points(xyz):
    ##open .xyz file
    f = open(xyz,"r")
    data = []
    ##get contents
    for line in f:
        data.append(line)
    ##delete first 2 lines which do not contain Cartesian coordinates
    del(data[0])
    del(data[0])
    ##make new lists to populate
    atom_pos=[]
    atom_names=[]
    atom_radii=[]
    ##for each line (atomic XYZ coordinates)
    for f in range(len(data)):
        ##remove line space
        data[f]=data[f].replace('\n','')
        ##split into atom,x,y,z
        data[f]=re.split(r'\s+',data[f])
        ##get atom name
        atom_names.append(data[f][0])
        ##delete atom name
        del(data[f][0])
    ##atom position is remaining x,y,z
    atom_pos=data
    ##convert coordinates to floats
    atom_pos = [[float(j) for j in i] for i in atom_pos]
    ##get the corresponding radius for each atom
    for f in range(len(atom_names)):
        atom_radii.append(radiiDict[atom_names[f]])
    ##define new list
    data=[]
    ##for each atom
    for f in range(len(atom_names)):
        ##Produce random points in a cube
        x=((2*atom_radii[f])*np.random.rand(number_points,3))-atom_radii[f]
        ##Keep points inside the sphere
        keep=[]
        for point in x:
            if math.sqrt(((point[0])**2)+((point[1])**2)+((point[2])**2)) < atom_radii[f]:
                keep.append(point)
        keep=np.array(keep)
        ##Project points to surface of sphere
        x1=[]
        y1=[]
        z1=[]
        for point in keep:
            d=math.sqrt(((point[0])**2)+((point[1])**2)+((point[2])**2))
            scale=(atom_radii[f]-d)/d
            point=point+(scale*point)
            x1.append(point[0])
            y1.append(point[1])
            z1.append(point[2])
        ##Move atom to correct position
        for i in range(len(x1)):
            x1[i]=x1[i]+atom_pos[f][0]
        for i in range(len(y1)):
            y1[i]=y1[i]+atom_pos[f][1]
        for i in range(len(z1)):
            z1[i]=z1[i]+atom_pos[f][2]
        data.append(x1)
        data.append(y1)
        data.append(z1)
    ##Discard points in shape interior
    for f in range(len(atom_names)):
        for g in range(len(atom_names)):
            if g==f:
                continue
            keep=[]
            for i in range(len(data[3*f])):
                if math.sqrt(((data[3*f][i]-atom_pos[g][0])**2)+((data[(3*f)+1][i]-atom_pos[g][1])**2)+((data[(3*f)+2][i]-atom_pos[g][2])**2)) > atom_radii[g]:
                    keep.append(i)
            x1_keep=[]
            y1_keep=[]
            z1_keep=[]
            for x in keep:
                x1_keep.append(data[3*f][x])
                y1_keep.append(data[(3*f)+1][x])
                z1_keep.append(data[(3*f)+2][x])
            data[(3*f)]=x1_keep
            data[(3*f)+1]=y1_keep
            data[(3*f)+2]=z1_keep
    x=[]
    y=[]
    z=[]
    ##merge points
    for f in range(len(data)):
        if f%3 == 0:
            for g in data[f]:
                x.append(g)
        if f%3 == 1:
            for g in data[f]:
                y.append(g)
        if f%3 == 2:
            for g in data[f]:
                z.append(g)
    ##return separate x, y and z point lists
    return(x,y,z)
##plot and save graphs method
def graph(x,y,z,az,el):
    fig = plt.figure(figsize=(20,20)) ##large canvas size for resolution and to fit larger molecules
    ##use 3d plotting
    ax = fig.add_subplot(111,projection='3d')
    ##colour black with big point size so image opaque
    ax.scatter(x,y,z,color="black",s=100)
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.set_zlim(-20,20)
    ##no axes!
    plt.axis('off')
    ##axim and alev are the angles to define the view, change to get projection down each axis
    ax.view_init(azim=az, elev=el)
	
##section 4: run method and save images
##save x, y, z surface points
for files in os.listdir(xyz_folder):
    ##get the points and save
    x,y,z = get_points(xyz_folder + "\\" + files)
    files=files.replace(".xyz","")
    f = open(vdw_points + "\\" + files + ".csv","a")
    for i in range(len(x)):
        f.write(str(x[i]) + "," + str(y[i]) + "," + str(z[i]) + "\n")
    f.close()
##reload surface points file and save images
for files in os.listdir(vdw_points):
    ##loads points
    data=np.loadtxt(vdw_points + "\\" + files,delimiter=",")
    files=files.replace(".csv","")
    ##get x, y, and z points
    x=data[:,0]
    y=data[:,1]
    z=data[:,2]
    ##get graphs and save
    graph(x,y,z,0,0)
    plt.savefig(image_folder + "\\" + files + "1.png") #first angle
    plt.close()
    graph(x,y,z,90,0)
    plt.savefig(image_folder + "\\" + files + "2.png") #perpendicular
    plt.close()
    graph(x,y,z,90,90)
    plt.savefig(image_folder + "\\" + files + "3.png") #perpendicular again
    plt.close()
