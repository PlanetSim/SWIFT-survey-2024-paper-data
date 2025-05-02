#SJL 11/15
#File containing the structures for dealing with HERCULES output

#import required modules
import numpy as np
import struct
import sys


###############################################################
###############################################################
###############################################################
#Parameters structure from HERCULES

class HERCULES_parameters(object):
    ###################################################
    def __init__(self):
        #Run name
        self.run_name="" 

        #max itterations and tollerance
        self.nint_max=0
        self.toll=0.0
        
        #max itterations and tollerance for xi calculation
        self.xi_nint_max=0
        self.xi_toll=0.0
        #xi step for differential calc
        self.dxi=0.0

        #flags and their parameters
        self.flag_start=0
        self.start_file=""
        self.flag_Mconc=0
        self.flag_Lconc=0
        self.flag_iter_print=0

        #rotational profile properties
        self.omega_param=np.empty(3)

    ###################################################
    #function to read the parameters from a HERCULES output
    def read_binary(self, f):
        self.run_name=f.read(200).rstrip()
        self.nint_max=struct.unpack('i', f.read(4))[0]
        self.toll=struct.unpack('d', f.read(8))[0]

        self.xi_nint_max=struct.unpack('i', f.read(4))[0]
        self.xi_toll=struct.unpack('d', f.read(8))[0]
        self.dxi=struct.unpack('d', f.read(8))[0]

        self.flag_start=struct.unpack('i', f.read(4))[0]
        self.start_file=f.read(200).rstrip()
        self.flag_Mconc=struct.unpack('i', f.read(4))[0]
        self.flag_Lconc=struct.unpack('i', f.read(4))[0]
        self.flag_iter_print=struct.unpack('i', f.read(4))[0]

        self.omega_param=np.asarray(struct.unpack('3d', f.read(3*8)))

        
###############################################################
###############################################################
###############################################################
#Planet structure from HERCULES

class HERCULES_planet(object):
    ###################################################
    def __init__(self):
        #Number of mu points, number of layers, number of different materials
        self.Nmu=0
        self.Nlayer=0
        self.Nmaterial=0
        
        #max legendre degree to calculate to
        self.kmax=0
        
        #Mass, AM of planet
        self.Mtot=0.0
        self.Ltot=0.0
        
        #various properties of planet
        self.omega_rot=0.0    #solid body rotation rate
        self.pmin=0.0         #pressure of outermost layer
        self.amax=0.0         #maximum equitorial radius
        self.aspect=0.0       #aspect ratio
        self.ref_rho=0.0      #constant density layers if needed
        self.Mtot_tar=0.0     #mass target for itteration
        self.Ltot_tar=0.0     #AM target for itteration
        self.Ucore=0.0        #Potential at the centre of body
        self.pcore=0.0        #pressure at centre of body

        #Vectors of real density, pressure, dpdr and equitorial potetnail of the layers
        self.real_rho=np.empty(0)
        self.press=np.empty(0)
        self.dpdr=np.empty(0)
        self.Ulayers=np.empty(0)
        self.Mout=np.empty(0)
        self.Lout=np.empty(0)

        #vector of Js
        self.Js=np.empty(0)

        #Vectors of angles
        self.mu=np.empty(0)

        #Vector of material flags
        self.flag_material=np.empty(0)
        
        #Vector of material masses
        self.M_materials=np.empty(0)

        #array of structures
        self.layers=[]

        #array of materials
        self.materials=[]

        #SJL 1/16
        #properties that we might want to know about planets
        self.pCMB=0.0
        self.pf_layer0=0.0

    ###################################################
    #function to read the structure from a HERCULES output
    def read_binary(self, f):
        #read ints
        self.Nmu=struct.unpack('i', f.read(4))[0]
        self.Nlayer=struct.unpack('i', f.read(4))[0]
        self.Nmaterial=struct.unpack('i', f.read(4))[0]
        self.kmax=struct.unpack('i', f.read(4))[0]

        #read doubles
        self.Mtot=struct.unpack('d', f.read(8))[0]
        self.Ltot=struct.unpack('d', f.read(8))[0]
        self.omega_rot=struct.unpack('d', f.read(8))[0]
        self.pmin=struct.unpack('d', f.read(8))[0]
        self.amax=struct.unpack('d', f.read(8))[0]
        self.aspect=struct.unpack('d', f.read(8))[0]
        self.ref_rho=struct.unpack('d', f.read(8))[0]
        self.Mtot_tar=struct.unpack('d', f.read(8))[0]
        self.Ltot_tar=struct.unpack('d', f.read(8))[0]
        self.Ucore=struct.unpack('d', f.read(8))[0]
        self.pcore=struct.unpack('d', f.read(8))[0]

        #read vectors
        self.real_rho=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
        self.press=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
        self.dpdr=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
        self.mu=np.asarray(struct.unpack(str(self.Nmu)+'d', f.read(self.Nmu*8)))
        self.Ulayers=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
        self.flag_material=np.asarray(struct.unpack(str(self.Nlayer)+'i', f.read(self.Nlayer*4)))
        self.M_materials=np.asarray(struct.unpack(str(self.Nmaterial)+'d', f.read(self.Nmaterial*8)))
        self.Mout=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
        self.Lout=np.asarray(struct.unpack(str(self.Nlayer)+'d', f.read(self.Nlayer*8)))
        self.Js=np.asarray(struct.unpack(str(self.kmax+1)+'d', f.read((self.kmax+1)*8)))


        #initialise layers and read in
        for i in np.arange(0,self.Nlayer):
            self.layers.append(HERCULES_concentric_layer())
            self.layers[i].read_binary(f)


        #initialise materials and read in
        for i in np.arange(0,self.Nmaterial):
            self.materials.append(HERCULES_EOS())
            self.materials[i].read_binary(f)

    ###################################################
    #function to calculate CMB pressure assuming the core is the lowest layer
    def calc_pCMB(self):
        core_layer=self.Nmaterial-1
        temp=np.where(self.flag_material==core_layer)[0]
        ind=temp[0]
        
        self.pCMB=self.press[ind]

    ###################################################
    #function to calculate CMB pressure assuming the core is the lowest layer
    #pass the fraction of material to find the pressure for, from core up
    def calc_pf_layer0(self, f):
        #redefine f for ease of use
        f=1-f

        #find an array of cumulative masses
        temp=np.asarray([0])
        for i in np.arange(self.Nlayer-1):
            temp=np.append(temp, ((self.layers[i].Vol-self.layers[i+1].Vol)*self.real_rho[i]))
        mass=np.cumsum(temp)

        #find the layer below the fraction f of the mantle
        temp=np.where(mass>f*self.M_materials[0])[0]
        ind_max=temp[0]

        #linearly interpolare
        frac=(f*self.M_materials[0]-mass[ind_max-1])/(mass[ind_max]-mass[ind_max-1])
        pf=self.press[ind_max]*frac+self.press[ind_max-1]*(1-frac)

        #print pf/1E9, self.press[ind_max]/1E9, self.press[ind_max-1]/1E9
        
        self.pf_layer0=pf
        

###############################################################
###############################################################
###############################################################
#Concentric layer structure from HERCULES
class HERCULES_concentric_layer(object):
    ###################################################
    def __init__(self):
        #number of mu points and max legendre degree
        self.Nmu=0
        self.kmax=0

        #the rotation rate of layer, density (only of concentric layer), mass 
        self.omega=0.0
        self.rho=0.0
        self.M=0.0
        self.Vol=0.0

        #equitorial and polar radius
        self.a=0.0 
        self.b=0.0
        
        #Moment of inertia
        self.I=0.0

        #Vector of the angles and xi
        self.mu=np.empty(0)
        self.xi=np.empty(0)

        #vector of Js
        self.Js=np.empty(0)

    ###################################################
    def read_binary(self, f):
        #read ints
        self.Nmu=struct.unpack('i', f.read(4))[0]
        self.kmax=struct.unpack('i', f.read(4))[0]

        #read doubles
        self.omega=struct.unpack('d', f.read(8))[0]
        self.rho=struct.unpack('d', f.read(8))[0]
        self.M=struct.unpack('d', f.read(8))[0]
        self.Vol=struct.unpack('d', f.read(8))[0]
        self.a=struct.unpack('d', f.read(8))[0]
        self.b=struct.unpack('d', f.read(8))[0]
        self.I=struct.unpack('d', f.read(8))[0]

        #read vectors
        self.mu=np.asarray(struct.unpack(str(self.Nmu)+'d', f.read(self.Nmu*8)))
        self.xi=np.asarray(struct.unpack(str(self.Nmu)+'d', f.read(self.Nmu*8)))
        self.Js=np.asarray(struct.unpack(str(self.kmax+1)+'d', f.read((self.kmax+1)*8)))


                        
###############################################################
###############################################################
###############################################################
#Concentric layer structure from HERCULES
class HERCULES_EOS(object):
    ###################################################
    def __init__(self):
        self.fname=""
        self.EOS_type=0


    ###################################################
    def read_binary(self, f):
        self.fname=f.read(200).rstrip()
        self.EOS_type=struct.unpack('i', f.read(4))[0]
        
