# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:41:06 2023

@author: adriana
"""
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('axes', labelsize=24)
matplotlib.rcParams["font.family"] = 'Times New Roman'
from matplotlib.colors import TwoSlopeNorm
 
import numpy as np
from scipy import interpolate
import unyt
from numba import njit
import h5py
import struct

import os,sys
from copy import deepcopy

import swiftsimio as sw
import woma
from woma.misc import utils, io
from woma.eos import tillotson, sesame, idg, hm80
from woma.eos.T_rho import T_rho
from woma.misc import glob_vars as gv
woma.load_eos_tables()
print(woma.__file__)

#this_dir, this_file = os.path.split(__file__)
#path = os.path.join(this_dir)
#sys.path.append(path)
#import gadget_sph
import yaml
#sys.path.append('/home/apostema/hercules/HERCULES_development/Tutorial_beta/Analysis_scripts')
sys.path.append('C:/Users/gerri/HERCULES_development/Tutorial_beta/Analysis_scripts/')
sys.path.append('/home/apostema/hercules/HERCULES_development/Tutorial_beta/Analysis_scripts/')
from HERCULES_structures import *

R_earth = gv.R_earth #6.371e6   # m
M_earth = gv.M_earth #5.9724e24  # kg 
G = gv.G #6.67408e-11  # m^3 kg^-1 s^-2
LEM=3.5E34 #AMof Earth-Moon system in mks

mant_mat_id = 403 #USER INPUT
core_mat_id = 402 #USER INPUT

def find_Lz(data,pos,vel,cm,vcm,index=None):
    '''
    Finds L for an input dataset around z-axis

    Parameters
    ----------
    data : swift dataset
    cm : initial center of mass offset - from find_initial_com
    vcm: initial velocity center of mass offset - from find_initial_com

    Returns
    -------
    Lz : float64
        Total angular momentum for input dataset - in LEM units

    '''
    if index is None:
        index = np.where(data.gas.particle_ids.value>=0)
    pos = pos[index]
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    #pos_z = pos.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    vel = vel[index]
    vel_x = vel.value[:,0] - vcm[0]
    vel_y = vel.value[:,1] - vcm[1]
    #vel_z = vel.value[:,2] - vcm[2]
    m = data.gas.masses.value[index]
    
    Lz = np.abs(np.sum(m * (pos_x*vel_y - pos_y*vel_x),dtype='float64'))
    return Lz/LEM

def find_Lz_particles(m,pos_x,pos_y,vel_x,vel_y,cm,vcm):
    '''
    Finds L for an input dataset around z-axis

    Parameters
    ----------
    data : swift dataset
    cm : initial center of mass offset - from find_initial_com
    vcm: initial velocity center of mass offset - from find_initial_com

    Returns
    -------
    Lz : float64
        Total angular momentum for input dataset - in LEM units

    '''
    #boxsize = data.metadata.boxsize.value
    #pos_x = pos.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    #pos_y = pos.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    #pos_z = pos.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    #vel = vel[index]
    #vel_x = vel.value[:,0] - vcm[0]
    #vel_y = vel.value[:,1] - vcm[1]
    #vel_z = vel.value[:,2] - vcm[2]
    #m = data.gas.masses.value[index]
    
    Lz = np.abs(np.sum(m * (pos_x*vel_y - pos_y*vel_x),dtype='float64'))
    return Lz/LEM

def pdt_solidus(P=None,T=None):
    pdt_arr=np.array([[0.0, 1706.3],
             [0.1, 1706.319],
             [2.3522, 1706.3197],
             [2.7285, 1780.6691],
             [3.4812, 1862.9846],
             [4.2339, 1939.9893],
             [5.1747, 2009.0281],
             [6.3038, 2088.6883],
             [7.621, 2157.7269],
             [9.0323, 2221.4551],
             [10.7258, 2269.2511],
             [12.9839, 2303.7706],
             [15.1478, 2322.3578],
             [16.1828, 2364.8433],
             [17.1237, 2402.0181],
             [18.1586, 2407.3287],
             [19.8521, 2409.9841],
             [21.5457, 2417.9501],
             [22.7688, 2441.8481],
             [23.8978, 2473.7122],
             [30.9543, 2648.9644],
             [35.0941, 2747.2119],
             [39.422, 2853.4254],
             [43.2796, 2941.0515],
             [46.1962, 3010.0903],
             [49.5833, 3079.129],
             [54.7581, 3187.9979],
             [58.0511, 3251.726],
             [64.2608, 3368.5608],
             [70.0941, 3474.7743],
             [77.8091, 3596.9198],
             [85.7124, 3719.0653],
             [91.6398, 3798.7254],
             [98.2258, 3878.3856],
             [105.0941, 3950.0796],
             [112.1505, 4021.7738],
             [118.7366, 4069.5697],
             [126.3575, 4125.3319],
             [131.9085, 4154.5406],
             [140.0, 4194.3707],
             [240.0, 4686.6168]])
    pdtsolidusPT=interpolate.interp1d(pdt_arr[:,0],pdt_arr[:,1])
    pdtsolidusTP=interpolate.interp1d(pdt_arr[:,1],pdt_arr[:,0])
    if P is not None:
        return pdtsolidusPT(P) #return temperature
    if T is not None:
        return pdtsolidusTP(T) #return pressure (in GPa)
    

def CMB_rubie(mplanet,mcore,mmantle):
    rhomant = 1063.83*mplanet + 3436.17
    rhocore = 2.5*rhomant
    return np.power(mcore/(4/3*np.pi*rhocore),1/3)
    
def P_rubie(r,rCMB,rplanet,mplanet,mcore,mmantle):
    rhomant = 1063.83*mplanet + 3436.17
    rhocore = 2.5*rhomant
    b = np.power(mcore/(4/3*np.pi*rhocore),1/3)
    a = np.power((mmantle+4/3*rhomant*np.pi*b**3)/(4/3*np.pi*rhomant),1/3)
    #a = np.power(mplanet*M_earth/(4/3*np.pi*(1-(rhomant/7/(rhocore/3+rhomant/7))*(rhocore-rhomant))),1/3)
    #b = rplanet*R_earth*np.power(4/7*rhomant/(4/3*rhocore + 4/7*rhomant),1/3)
    #a = rplanet*R_earth
    #b = rCMB*R_earth
    if (r<=a and r>b):
        return 4/3*np.pi*rhomant*G*b**3*(rhocore-rhomant)*(1/r-1/a) + 2/3*np.pi*G*rhomant**2*(a**2-r**2)
    if (r>0 and r<=b):
        return 2/3*np.pi*G*rhocore**2*(b**2-r**2) + 2/3*np.pi*G*rhomant**2*(a**2-b**2) + 4/3*np.pi*rhomant*G*b**3*(rhocore-rhomant)*(1/b-1/a)
    
def T_rubie(P):
    P=P/1.e9
    if P<24:
        return 1874 + 55.43*P - 1.74*P**2 + 0.0193*P**3
    if P>=24:
        return 1249 + 58.28*P - 0.395*P**2 + 0.0011*P**3

def find_initial_com(data):
    '''
    Finds inital center of mass of the entire system at time zero

    Parameters
    ----------
    data : swift dataset
        data.gas.coordinates: contains x/y/z coordinates of each particle - originally + 40 R_earth.
        data.gas.velocities: contains x/y/z velocities of each particle - originally + 40 R_earth.
        data.gas.masses: masses of each particle 

    Returns
    -------
    cm : [xcm,ycm,zcm] : coordinate offset of initial center of mass (not offset by 40 R_earth!)
    vcm : [vxcm,vycm,vzcm] : velocities of initial center of mass
    
    '''
    #stolen from gadget_sph.py from SJL 1/11/16
    Ncenter = 200
   
    mat_id = data.gas.material_ids.value
    part_id = data.gas.particle_ids.value
    pos = data.gas.coordinates
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth
    vel = data.gas.velocities
    vel_x = vel.value[:,0]
    vel_y = vel.value[:,1]
    vel_z = vel.value[:,2]
    m = data.gas.masses.value
    pot = data.gas.potentials.value
    
    mtot = np.sum(m)
    xcm = np.sum(pos_x*m)/mtot
    ycm = np.sum(pos_y*m)/mtot
    zcm = np.sum(pos_z*m)/mtot
    cm = [xcm,ycm,zcm] #not including 40 R_earth offset!
    vxcm = np.sum(vel_x*m)/mtot
    vycm = np.sum(vel_y*m)/mtot
    vzcm = np.sum(vel_z*m)/mtot
    vcm = [vxcm,vycm,vzcm] 
    
    return cm,vcm

def apply_index(data,index):
    '''
    Applies an index (swift mask) to an existing swift dataset and returns a dataset containing just those particles

    Parameters
    ----------
    data : particles
        Dataset to take a subset of defined by index
    index : array-like
        Np.where index from which to return a dataset.

    Returns
    -------
    data2 : particles
        swift dataset of just the particles in the specified index

    '''
    mask = sw.SWIFTMask(data.metadata,spatial_only=False)
    mask.constrain_mask('gas', 'particle_ids', -2, -1) #set all values to false using built-in method
    mask.gas[index] = 1 # manually set index
    
    data2 = sw.load(data.metadata.filename,mask=mask)
    data2.gas.coordinates.convert_to_mks()
    data2.gas.pressures.convert_to_mks()
    data2.gas.internal_energies.convert_to_mks()
    data2.gas.masses.convert_to_mks()
    data2.gas.velocities.convert_to_mks()
    data2.gas.smoothing_lengths.convert_to_mks()
    data2.gas.densities.convert_to_mks()
    data2.gas.potentials.convert_to_mks()
    
    return data2

def bound_mass(data,cm,vcm):
    '''
    Finds bound core particles CoM, then finds the bound mass around those particles - not the same as total CoM

    Parameters
    ----------
    data : swift dataset
        Swift dataset, does not need to be adjusted to any CoM frame
    cm : center of mass offset - from find_initial_com
    vcm: velocity center of mass offset - from find_initial_com

    Returns
    -------
    indbnd : np.array index
        Index of bound particles
    newm/M_earth : float64
        Final bound mass in earth units
    pos : np.array of swift data position values adjusted to the bound center of mass

    '''
    #stolen from gadget_sph.py from SJL 1/11/16
    Ncenter = 200
    #this function finds the center of mass of the mantle particles with lowest PE
    #returns all of the data, with coordinate positions and velocities shifted
    mat_id = data.gas.material_ids.value
    part_id = data.gas.particle_ids.value
    pos = data.gas.coordinates
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    vel = data.gas.velocities
    vel_x = vel.value[:,0] - vcm[0]
    vel_y = vel.value[:,1] - vcm[1]
    vel_z = vel.value[:,2] - vcm[2]
    m = data.gas.masses.value
    pot = data.gas.potentials.value
    
    indmantle = np.where(mat_id == mant_mat_id)
    indcore = np.where(mat_id == core_mat_id)
    
    coreSort = np.argsort(pot[indcore])
    xcm = np.mean(pos_x[indcore][coreSort[0:Ncenter]])
    ycm = np.mean(pos_y[indcore][coreSort[0:Ncenter]])
    zcm = np.mean(pos_z[indcore][coreSort[0:Ncenter]])
    vxcm = np.mean(vel_x[indcore][coreSort[0:Ncenter]])
    vycm = np.mean(vel_y[indcore][coreSort[0:Ncenter]])
    vzcm = np.mean(vel_z[indcore][coreSort[0:Ncenter]])
    bndm = np.sum(m[indcore][coreSort])
    oldBoundPot = np.mean(pot[indcore][coreSort[0:Ncenter]])
    print('center: ', xcm/R_earth, ycm/R_earth, zcm/R_earth)
    
    #iterating - still not entirely confident about what's going on here - update: maybe more confident
    #testing this with just bound core mass at the moment? might see how things differ
    bndflag=np.zeros(np.shape(m)[0])
    newm = bndm/10.0
    tolerance = 1.E-3 #or just choose another value idk
    nn=0
    while(np.absolute((newm-bndm)/newm)>tolerance and nn<50):
        print(nn)
        nn=nn+1
        bndm = newm
        #finding KE and PE of all particles
        v2 = np.power(vel_x-vxcm,2.0) + np.power(vel_y-vycm,2.0) + np.power(vel_z-vzcm,2.0)
        r = np.sqrt(np.power(pos_x-xcm,2.0) + np.power(pos_y-ycm,2.0) + np.power(pos_z-zcm,2.0))
        KE = 0.5 * m * v2
        PE = -1*G * bndm * np.divide(m,r)
        
        indbnd = np.where(KE+PE < 0.0)
        bndflag[indbnd]=1
        indesc = np.where(KE+PE >= 0.0)
        bndflag[indesc]=0

        print('Nbound: ',np.shape(indbnd)[1])
        
        #recalculate bound and escaping mass
        newm = np.sum(m[indbnd])
        escm = np.sum(m[indesc])

        indbndCore = np.where((mat_id == core_mat_id) & (KE+PE < 0.0))
        boundSort = np.argsort(pot[indbndCore])
        #newBoundPot = np.mean(PE[boundSort[0:Ncenter]])
        newBoundPot = np.mean(pot[indbndCore][boundSort[0:Ncenter]])
        print(oldBoundPot)
        #reset center to lowest Ncenter potentials of bound core mass
        if (np.shape(indbndCore)[1] > Ncenter and newBoundPot<oldBoundPot):
            oldBoundPot = newBoundPot
            xcm = np.mean(pos_x[indbndCore][boundSort[0:Ncenter]])
            ycm = np.mean(pos_y[indbndCore][boundSort[0:Ncenter]])
            zcm = np.mean(pos_z[indbndCore][boundSort[0:Ncenter]])
            vxcm = np.mean(vel_x[indbndCore][boundSort[0:Ncenter]])
            vycm = np.mean(vel_y[indbndCore][boundSort[0:Ncenter]])
            vzcm = np.mean(vel_z[indbndCore][boundSort[0:Ncenter]])
        
        print('bound mass: ',bndm/M_earth,' | new mass: ',newm/M_earth,' | diff: ',np.absolute(newm-bndm)/M_earth)
        print('center: ', xcm/R_earth, ycm/R_earth, zcm/R_earth)
        
    #moving planet back into central frame (though still at the center of the box)
    pos2=deepcopy(pos)
    vel2=deepcopy(vel)
    for i in range(len(m)):
        pos2[i,0] = pos_x[i] - xcm + boxsize[0]/2*R_earth
        pos2[i,1] = pos_y[i] - ycm + boxsize[1]/2*R_earth
        pos2[i,2] = pos_z[i] - zcm + boxsize[2]/2*R_earth
        vel2[i,0] = vel_x[i] - vxcm
        vel2[i,1] = vel_y[i] - vycm
        vel2[i,2] = vel_z[i] - vzcm
        
    return indbnd, newm/M_earth,pos2,vel2

def find_disk(data,cm,vcm,pos,vel,basename='',rplanetmax=None):
    '''
    Separate bound masses into planet and disk.

    Parameters
    ----------
    data : swift dataset
        Swift dataset, does not need to be adjusted to any CoM frame
        SHOULD BE THE BOUND DATASET MASK ONLY
    cm : initial center of mass offset - from find_initial_com
    vcm: initial velocity center of mass offset - from find_initial_com

    Returns
    -------
    inddisk : np.array index
        Index of bound disk particles
    indplanet : np.array index
        Index of bound planet particles
    diskm/M_earth : float64
        Final disk mass in earth units
    planetm/M_earth : float64
        Final planet mass in earth units

    '''
    # moving CoM/VCoM of BOUND particles - for calculating planet radius, NOT ANGULAR MOMENTUM
    bcm,bvcm = find_initial_com(data)
    #for i in range(3):
        #bcm[i]-=cm[i]
        #bvcm[i]-=vcm[i]
    print('Bound CoM/vCoM: ',bcm,bvcm)
    
    mat_id = data.gas.material_ids.value
    part_id = data.gas.particle_ids.value
    data.gas.coordinates.convert_to_mks()
    #pos = data.gas.coordinates
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth #- cm[0]
    pos_y = pos.value[:,1]-boxsize[0]/2*R_earth #- cm[1]
    pos_z = pos.value[:,2]-boxsize[0]/2*R_earth #- cm[2]
    rrr = np.sqrt((pos_x)**2 + (pos_y)**2 + (pos_z)**2)
    #vel = data.gas.velocities
    vel_x = vel.value[:,0] #- vcm[0]
    vel_y = vel.value[:,1] #- vcm[1]
    vel_z = vel.value[:,2] #- vcm[2]
    v2 = np.power(vel_x,2.0) + np.power(vel_y,2.0) + np.power(vel_z,2.0)
    vvv = np.sqrt(v2)
    m = data.gas.masses.value
    KE = 0.5 * m * v2
    pot = data.gas.potentials.value
    Lz = np.abs(m*(pos_x*vel_y - pos_y*vel_x))
    
    # Using a moving average calculation to find the planet-disk boundary (max particle KE)
    if rplanetmax is None:
        rplanetmax=np.amax(rrr)
    Rsort = np.argsort(rrr)
    Navg = 500
    KEmax = 0
    KEmin = np.amax(KE)
    rKEmax = 0
    rKEmin = 0
    KEavglist = np.zeros(len(rrr)-Navg-1)
    KEdifflist = np.zeros(len(rrr)-Navg-2)
    #vvvavglist = np.zeros(len(rrr)-Navg-1)
    rcutoff = rrr[Rsort[-1]]
    kecutoff = rrr[Rsort[-1]]
    rfirstpeak = 0
    i=0
    #for i in range(len(rrr)-Navg-1):
    while rrr[Rsort[i+Navg+1]]<rplanetmax:
        KEavg = np.sum(KE[Rsort[i:i+Navg]])/Navg
        #vvvavg = np.sum(vvv[Rsort[i:i+Navg]])/Navg
        KEavglist[i]=KEavg
        if i>0: 
            KEdifflist[i-1]=KEavg - KEavglist[i-1]
        #vvvavglist[i] = vvvavg
        ravg = np.sum(rrr[Rsort[i+1:i+Navg]]-rrr[Rsort[i:i+Navg-1]])/(Navg-1)
        KEdiff = np.sum(KE[Rsort[i+1:i+Navg]]-KE[Rsort[i:i+Navg-1]])/(Navg-1)
        if KEavg > KEmax:
            KEmax = KEavg
            rKEmax = rrr[Rsort[i]]
            KEmin = KEmax
        else:
            if KEavg < KEmin:
                rKEmin = rrr[Rsort[i]]
        if (rrr[Rsort[i+Navg+1]]-rrr[Rsort[i+Navg]]) > 1000*ravg:
            rcutoff = np.amin([rrr[Rsort[i]],rcutoff])
        if (KE[Rsort[i+Navg+1]]-KE[Rsort[i+Navg]]) > 1000*KEdiff:
            kecutoff = np.amin([rrr[Rsort[i]],kecutoff])
        i+=1
    rdvpeak = rrr[Rsort[np.argmin(KEdifflist)+Navg-2]]
    irdvpeak = np.argmin(KEdifflist)+Navg-2
    while rdvpeak >  (np.std(rrr)*2):
        rdvpeak = rrr[Rsort[np.argmin(KEdifflist[:irdvpeak-Navg+2])]]
        irdvpeak = np.argmin(KEdifflist[:irdvpeak])-Navg+2
    iKEdvpeak = np.argmax(KEavglist[:irdvpeak])
    rKEdvpeak = rrr[Rsort[iKEdvpeak]]
    print ('cutoff values: ',rKEmax/R_earth,rcutoff/R_earth,rKEmin/R_earth,rdvpeak/R_earth,rKEdvpeak/R_earth)
    #if rcutoff > kecutoff:
        #rcutoff = kecutoff
    rplanet = np.amin([rKEmax,rKEdvpeak])
    #print(rplanet/R_earth)
    if (rdvpeak < rKEmax < (1.03*rdvpeak)) & (rKEdvpeak < rdvpeak < (1.03*rKEdvpeak)): #sometimes the above misses outer layer of very defined planets
        rplanet = rKEmax
    meanmat = np.mean(mat_id[np.where(rrr<rplanet)])
    #print('Mean mat id: ',meanmat)
    #check to see if we accidentally ended up at the CMB - in this case planets are well-formed so we take overall KE peak
    if mant_mat_id>core_mat_id:
        if meanmat < (.95*core_mat_id+.05*mant_mat_id):
            rplanet = rKEmax
    if mant_mat_id<core_mat_id:
        if meanmat > (.95*core_mat_id+.05*mant_mat_id):
            rplanet = rKEmax
    inddisk = np.where(rrr > rplanet)
    indplanet = np.where(rrr <= rplanet)
    diskm = np.sum(m[inddisk])
    planetm = np.sum(m[indplanet])
    
    print('Planet/disk boundary: r = ',rplanet/R_earth,'R_earth')
    print('Planet mass: ',planetm/M_earth,' fraction of bound mass: ',planetm/np.sum(m))
    print('Disk mass: ',diskm/M_earth,' fraction of bound mass: ',diskm/np.sum(m))
    
    fir, ax = plt.subplots(1, 5, figsize=(20,4))
    
    zlim = np.std(pos_z)/3
    indmantle = np.where(mat_id == mant_mat_id)
    indcore = np.where(mat_id == core_mat_id)
    indcoreEq = np.where((np.absolute(pos_z) < (zlim)) & (mat_id == core_mat_id))
    indmantleEq = np.where((np.absolute(pos_z) < (zlim)) & (mat_id == mant_mat_id))
    
    rlim = np.std(rrr)*5/R_earth
    #ax[0].scatter(rrr/R_earth, vvv,s=2)
    ax[0].scatter(rrr[indmantle] / R_earth, vvv[indmantle],s=2)
    ax[0].scatter(rrr[indcore] / R_earth, vvv[indcore],s=2)
    ax[0].vlines(rKEmax/R_earth,0,np.amax(vvv),color='black')
    ax[0].vlines(rdvpeak/R_earth,0,np.amax(vvv),color='red')
    ax[0].vlines(rKEdvpeak/R_earth,0,np.amax(vvv),color='blue')
    ax[0].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[0].set_ylabel(r"Velocities, [m/s]")
    ax[0].set_xlim(0, rlim)
    ax[0].set_ylim(0, None)
    
    ax[1].scatter(rrr[indmantle] / R_earth, KE[indmantle],s=2)
    ax[1].scatter(rrr[indcore] / R_earth, KE[indcore],s=2)
    ax[1].vlines(rKEmax/R_earth,0,np.amax(KE),color='black')
    ax[1].vlines(rdvpeak/R_earth,0,np.amax(KE),color='red')
    ax[1].vlines(rKEdvpeak/R_earth,0,np.amax(KE),color='blue')
    ax[1].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[1].set_ylabel(r"Kinetic energy, [J]")
    ax[1].set_xlim(0, rlim)
    ax[1].set_ylim(0, None)
    
    ax[2].scatter(rrr[indmantle] / R_earth, Lz[indmantle],s=2)
    ax[2].scatter(rrr[indcore] / R_earth, Lz[indcore],s=2)
    ax[2].vlines(rKEmax/R_earth,0,np.amax(Lz),color='black')
    ax[2].vlines(rdvpeak/R_earth,0,np.amax(Lz),color='red')
    ax[2].vlines(rKEdvpeak/R_earth,0,np.amax(Lz),color='blue')
    ax[2].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[2].set_ylabel(r"Angular Momentum, $r$ $[kg m^2 s^-1]$")
    ax[2].set_xlim(0, rlim)
    ax[2].set_ylim(0, None)
    
    ax[3].plot(rrr[Rsort[:-Navg-1]] / R_earth, KEavglist)
    ax[3].vlines(rKEmax/R_earth,0,np.amax(KE),color='black')
    ax[3].vlines(rdvpeak/R_earth,0,np.amax(KE),color='red')
    ax[3].vlines(rKEdvpeak/R_earth,0,np.amax(KE),color='blue')
    ax[3].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[3].set_ylabel(r"Moving kinetic energy average, [J]")
    ax[3].set_xlim(0, rlim)
    ax[3].set_ylim(0, None)
    
    ax[4].plot(rrr[Rsort[1:-Navg-1]] / R_earth, KEdifflist)
    ax[4].vlines(rKEmax/R_earth,np.amin(KEdifflist),np.amax(KEdifflist),color='black')
    ax[4].vlines(rdvpeak/R_earth,np.amin(KEdifflist),np.amax(KEdifflist),color='red')
    ax[4].vlines(rKEdvpeak/R_earth,np.amin(KEdifflist),np.amax(KEdifflist),color='blue')
    ax[4].vlines(np.std(rrr)*2/R_earth,np.amin(KEdifflist),np.amax(KEdifflist),color='green')
    ax[4].set_xlabel(r"Radius, $r$ $[R_\oplus]$")
    ax[4].set_ylabel(r"Moving velocity average derivative")
    ax[4].set_xlim(0, rlim)
    ax[4].set_ylim(None, None)
    
    plt.tight_layout()
    imname = basename+'_R_disk_test.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    return inddisk, indplanet, diskm/M_earth, planetm/M_earth, rplanet/R_earth
    

def find_energy(data,cm,vcm):
    '''
    Finds total internal energy, kinetic energy, and gravitational potential energy

    Parameters
    ----------
    data : swift dataset
    cm : center of mass offset - from find_initial_com
    vcm: velocity center of mass offset - from find_initial_com

    Returns
    -------
    KEtot,IEtot,GPEtot,GPEmin
        KEtot : Total system kinetic energy (J)
        IEtot : total system internal energy (J)
        GPEtot : total system gravitational potential energy (J)
        GPEmin : minimum gravitational potential energy of dataset (J)

    '''
    pos = data.gas.coordinates
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    vel = data.gas.velocities
    vel_x = vel.value[:,0] - vcm[0]
    vel_y = vel.value[:,1] - vcm[1]
    vel_z = vel.value[:,2] - vcm[2]
    m = data.gas.masses.value
    pot = data.gas.potentials.value
    u = data.gas.internal_energies.value
    
    v2 = np.power(vel_x,2.0) + np.power(vel_y,2.0) + np.power(vel_z,2.0)
    ke = 0.5*m*v2
    KEtot = np.sum(ke)
    ie = u*m
    IEtot = np.sum(ie)
    gpe = pot*m * 0.5 #correct for extra factor of two
    GPEtot = np.sum(gpe)
    GPEmin = np.amin(gpe)
    
    return KEtot,IEtot,GPEtot,GPEmin

def HERCULES_profiles(data,cm,vcm,pos,vel,index=None,basename=''):
    if basename != '':
        basename = basename+'_'
    boxsize = data.metadata.boxsize.value
    mat_id = data.gas.material_ids.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth #- cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth #- cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth #- cm[2]
    vel_x = vel.value[:,0] #- vcm[0]
    vel_y = vel.value[:,1] #- vcm[1]
    vel_z = vel.value[:,2] #- vcm[2]
    m = data.gas.masses.value
    pot = data.gas.potentials.value/2
    u = data.gas.internal_energies.value
    rho = data.gas.densities.value
    #Lz = np.abs(m*(pos_x*vel_y - pos_y*vel_x))
    Lz = (m*(pos_x*vel_y - pos_y*vel_x))
    v2 = np.power(vel_x,2.0) + np.power(vel_y,2.0) + np.power(vel_z,2.0)
    ke = 0.5*m*v2
    tmp_s = np.zeros(np.size(u))
    #tmp_T = np.zeros(np.size(u))
    for i in range(np.size(u)):
        tmp_s[i] = sesame.s_u_rho(u[i],rho[i],mat_id[i])
        #tmp_T[i] = sesame.T_rho_s(rho[i],tmp_s[i],mat_id[i])
    rrr = np.sqrt((pos_x)**2 + (pos_y)**2 + (pos_z)**2)
    rr = np.sqrt((pos_x)**2 + (pos_y)**2)
    if index is None:
        index=np.where(m>0)
    indmant = np.where(mat_id[index]==mant_mat_id)
    indcore = np.where(pot[index]<np.amin(pot[index][indmant]))
    
    potSortcore = np.argsort(pot[index][indcore]*m[index][indcore])
    potSortmant = np.argsort(pot[index][indmant]*m[index][indmant])
    
    fraclistcore = np.zeros(102)
    masslistcore = np.zeros(102)
    Slistcore = np.zeros(102)
    rholistcore = np.zeros(102)
    potlistcore = np.zeros(102)
    ulistcore = np.zeros(102)
    KElistcore = np.zeros(102)
    Etotlistcore = np.zeros(102)
    
    for i in range(100):
        fraclistcore[i+1]=0.005+i*0.01
        tempind = np.where(((pot[index][indcore]*m[index][indcore])>np.percentile(pot[index][indcore]*m[index][indcore],i)) & ((pot[index][indcore]*m[index][indcore])<=np.percentile(pot[index][indcore]*m[index][indcore],i+1)))
        masslistcore[i+1] = np.sum(m[index][indcore][tempind],dtype='float64')
        Slistcore[i+1] = np.mean(tmp_s[index][indcore][tempind])#*m[index][indcore][tempind])
        rholistcore[i+1] = np.mean(rho[index][indcore][tempind])
        potlistcore[i+1] = np.mean(pot[index][indcore][tempind],dtype='float64')#*m[index][indcore][tempind],
        ulistcore[i+1] = np.mean(u[index][indcore][tempind],dtype='float64')#*m[index][indcore][tempind]
        KElistcore[i+1] = np.mean(ke[index][indcore][tempind]/m[index][indcore][tempind],dtype='float64')
        Etotlistcore[i+1] = potlistcore[i+1]+ulistcore[i+1]+KElistcore[i+1]
    
    fraclistcore[0]=0
    fraclistcore[101]=1
    masslistcore[0]=masslistcore[1]
    masslistcore[101]=masslistcore[100]
    Slistcore[0]=Slistcore[1]
    Slistcore[101]=Slistcore[100]
    rholistcore[0]=rholistcore[1]
    rholistcore[101]=rholistcore[100]
    potlistcore[0]=potlistcore[1]
    potlistcore[101]=potlistcore[100]
    ulistcore[0]=ulistcore[1]
    ulistcore[101]=ulistcore[100]
    KElistcore[0]=KElistcore[1]
    KElistcore[101]=KElistcore[100]
    Etotlistcore[0]=Etotlistcore[1]
    Etotlistcore[101]=Etotlistcore[100]
    
    with open(basename+'HERCULES_core_profiles.csv','w') as writefile:
        writefile.write('0'+'\n'+basename+''+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n')
        writefile.write('Mass(kg),Entropy(kg(K^-1)(m/s)^2),Densisty(kg/m^3),Potential Energy(kg(m/s)^2),Internal Energy(kg(m/s)^2),Kinetic Energy(kg(m/s)^2),Total Av. Energy(kg(m/s)^2)'+'\n')
        for i in range(102):
            writefile.write('{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}'.format(fraclistcore[i],Slistcore[i],rholistcore[i],potlistcore[i],ulistcore[i],KElistcore[i],Etotlistcore[i])+'\n')
    print('Saved file ',basename+'HERCULES_core_profiles.csv')
    
    fraclistmant = np.zeros(102)
    masslistmant = np.zeros(102)
    Slistmant = np.zeros(102)
    rholistmant = np.zeros(102)
    potlistmant = np.zeros(102)
    ulistmant = np.zeros(102)
    KElistmant = np.zeros(102)
    Etotlistmant = np.zeros(102)
    
    for i in range(100):
        fraclistmant[i+1]=0.005+i*0.01
        tempind = np.where(((pot[index][indmant]*m[index][indmant])>np.percentile(pot[index][indmant]*m[index][indmant],i)) & ((pot[index][indmant]*m[index][indmant])<=np.percentile(pot[index][indmant]*m[index][indmant],i+1)))
        masslistmant[i+1] = np.sum(m[index][indmant][tempind],dtype='float64')
        Slistmant[i+1] = np.mean(tmp_s[index][indmant][tempind])#*m[index][indmant][tempind])
        rholistmant[i+1] = np.mean(rho[index][indmant][tempind])
        potlistmant[i+1] = np.mean(pot[index][indmant][tempind],dtype='float64')#*m[index][indmant][tempind],
        ulistmant[i+1] = np.mean(u[index][indmant][tempind],dtype='float64')#*m[index][indmant][tempind]
        KElistmant[i+1] = np.mean(ke[index][indmant][tempind]/m[index][indmant][tempind],dtype='float64')
        Etotlistmant[i+1] = potlistmant[i+1]+ulistmant[i+1]+KElistmant[i+1]
    
    fraclistmant[0]=0
    fraclistmant[101]=1
    masslistmant[0]=masslistmant[1]
    masslistmant[101]=masslistmant[100]
    Slistmant[0]=Slistmant[1]
    Slistmant[101]=Slistmant[100]
    rholistmant[0]=rholistmant[1]
    rholistmant[101]=rholistmant[100]
    potlistmant[0]=potlistmant[1]
    potlistmant[101]=potlistmant[100]
    ulistmant[0]=ulistmant[1]
    ulistmant[101]=ulistmant[100]
    KElistmant[0]=KElistmant[1]
    KElistmant[101]=KElistmant[100]
    Etotlistmant[0]=Etotlistmant[1]
    Etotlistmant[101]=Etotlistmant[100]
    
    with open(basename+'HERCULES_mant_profiles.csv','w') as writefile:
        writefile.write('0'+'\n'+basename+''+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n')
        writefile.write('Mass(kg),Entropy(kg(K^-1)(m/s)^2),Densisty(kg/m^3),Potential Energy(kg(m/s)^2),Internal Energy(kg(m/s)^2),Kinetic Energy(kg(m/s)^2),Total Av. Energy(kg(m/s)^2)'+'\n')
        for i in range(102):
            writefile.write('{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}'.format(fraclistmant[i],Slistmant[i],rholistmant[i],potlistmant[i],ulistmant[i],KElistmant[i],Etotlistmant[i])+'\n')
    print('Saved file ',basename+'HERCULES_mantle_profiles.csv')
    
    fir, ax = plt.subplots(2, 3, figsize=(18,12))
    ax[0,0].scatter(rr[index][indmant]/R_earth,np.abs(pos_z[index][indmant])/R_earth,c=rho[index][indmant],s=2,cmap='inferno')
    ax[0,0].scatter(rr[index][indcore]/R_earth,np.abs(pos_z[index][indcore])/R_earth,c=rho[index][indcore],s=2,cmap='pink')
    ax[0,0].set_xlabel('Radius [R_earth]')
    ax[0,0].set_ylabel('Height [R_earth]')
    ax[0,0].set_title('Density')
    
    ax[0,1].scatter(rr[index][indmant]/R_earth,np.abs(pos_z[index][indmant])/R_earth,c=tmp_s[index][indmant],s=2,cmap='inferno')
    ax[0,1].scatter(rr[index][indcore]/R_earth,np.abs(pos_z[index][indcore])/R_earth,c=tmp_s[index][indcore],s=2,cmap='pink')
    ax[0,1].set_xlabel('Radius [R_earth]')
    ax[0,1].set_ylabel('Height [R_earth]')
    ax[0,1].set_title('Entropy')
    
    ax[0,2].scatter(rr[index][indmant]/R_earth,np.abs(pos_z[index][indmant])/R_earth,c=pot[index][indmant],s=2,cmap='inferno')
    ax[0,2].scatter(rr[index][indcore]/R_earth,np.abs(pos_z[index][indcore])/R_earth,c=pot[index][indcore],s=2,cmap='pink')
    ax[0,2].set_xlabel('Radius [R_earth]')
    ax[0,2].set_ylabel('Height [R_earth]')
    ax[0,2].set_title('Potential Energy')
    
    ax[1,0].scatter(rr[index][indmant]/R_earth,np.abs(pos_z[index][indmant])/R_earth,c=u[index][indmant],s=2,cmap='inferno')
    ax[1,0].scatter(rr[index][indcore]/R_earth,np.abs(pos_z[index][indcore])/R_earth,c=u[index][indcore],s=2,cmap='pink')
    ax[1,0].set_xlabel('Radius [R_earth]')
    ax[1,0].set_ylabel('Height [R_earth]')
    ax[1,0].set_title('Internal Energy')
    
    ax[1,1].scatter(rr[index][indmant]/R_earth,np.abs(pos_z[index][indmant])/R_earth,c=ke[index][indmant]/m[index][indmant],s=2,cmap='inferno')
    ax[1,1].scatter(rr[index][indcore]/R_earth,np.abs(pos_z[index][indcore])/R_earth,c=ke[index][indcore]/m[index][indcore],s=2,cmap='pink')
    ax[1,1].set_xlabel('Radius [R_earth]')
    ax[1,1].set_ylabel('Height [R_earth]')
    ax[1,1].set_title('Kinetic Energy')
    
    ax[1,2].scatter(rr[index][indmant]/R_earth,np.abs(pos_z[index][indmant])/R_earth,c=pot[index][indmant]+u[index][indmant]+ke[index][indmant]/m[index][indmant],s=2,cmap='inferno')
    ax[1,2].scatter(rr[index][indcore]/R_earth,np.abs(pos_z[index][indcore])/R_earth,c=pot[index][indcore]+u[index][indcore]+ke[index][indcore]/m[index][indcore],s=2,cmap='pink')
    ax[1,2].set_xlabel('Radius [R_earth]')
    ax[1,2].set_ylabel('Height [R_earth]')
    ax[1,2].set_title('Total Energy')
    
    plt.tight_layout()
    imname=basename+'_HERCULES_profiles.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    ## CALCULATING RUBIE HERCULES PROFILES
    
    mcore = np.sum(m[index][indcore],dtype='float64')
    mmantle = np.sum(m[index][indmant],dtype='float64')
    rhomant = 1063.83*(mcore+mmantle)/M_earth + 3436.17
    rhocore = 2.5*rhomant
    
    rCMBrubie = CMB_rubie((mcore+mmantle)/M_earth, mcore, mmantle)
    print('Rubie CMB: ',rCMBrubie/R_earth,' [R_earth]')
    PCMBrubie = P_rubie(rCMBrubie, 0, 0, (mcore+mmantle)/M_earth, mcore, mmantle)
    print('PCMB Rubie: ',PCMBrubie/1.e9,' GPa')
    #TCMBrubie = pdt_solidus(P=PCMBrubie/1.e9)
    TCMBrubie = T_rubie(PCMBrubie)
    SCMBrubie = sesame.s_rho_T(rhocore,TCMBrubie,core_mat_id)
    print('SCMB Rubie ',SCMBrubie/1.e3,' kJ/K/kg')
    r0=0
    #r0=3/4/np.pi/rhocore*(mcore/100)
    
    fraclistRubiecore = np.zeros(102)
    #masslistRubiecore = np.zeros(102)
    SlistRubiecore = np.zeros(102)
    rholistRubiecore = np.zeros(102)
    potlistRubiecore = np.zeros(102)
    ulistRubiecore = np.zeros(102)
    KElistRubiecore = np.zeros(102)
    EtotlistRubiecore = np.zeros(102)
    
    for i in range(100):
        fraclistRubiecore[i+1]=0.005+i*0.01
        r1 = (3/4/np.pi/rhocore*(mcore/100)+r0**3)**(1/3)
        player = P_rubie((r1+r0)/2, 0, 0, (mcore+mmantle)/M_earth, mcore, mmantle)
        rholistRubiecore[i+1] = rhocore
        SlistRubiecore[i+1] = SCMBrubie #sesame.Z_rho_Y(float(rhocore),float(player),int(core_mat_id),str("s"),str("P"))
        potlistRubiecore[i+1] = -G*(mcore/100)*(i+1)/((r1+r0)/2)
        ulistRubiecore[i+1] = sesame.Z_rho_Y(rhocore,player,core_mat_id,'u','P')
        KElistRubiecore[i+1] = 1.
        EtotlistRubiecore[i+1] = potlistRubiecore[i+1]+ulistRubiecore[i+1]
        r0=r1
    
    fraclistRubiecore[0]=0
    fraclistRubiecore[101]=1
    SlistRubiecore[0]=SlistRubiecore[1]
    SlistRubiecore[101]=SlistRubiecore[100]
    rholistRubiecore[0]=rholistRubiecore[1]
    rholistRubiecore[101]=rholistRubiecore[100]
    potlistRubiecore[0]=potlistRubiecore[1]
    potlistRubiecore[101]=potlistRubiecore[100]
    ulistRubiecore[0]=ulistRubiecore[1]
    ulistRubiecore[101]=ulistRubiecore[100]
    KElistRubiecore[0]=KElistRubiecore[1]
    KElistRubiecore[101]=KElistRubiecore[100]
    EtotlistRubiecore[0]=EtotlistRubiecore[1]
    EtotlistRubiecore[101]=EtotlistRubiecore[100]
    
    with open(basename+'HERCULES_Rubie_core_profiles.csv','w') as writefile:
        writefile.write('0'+'\n'+basename+''+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n')
        writefile.write('Mass(kg),Entropy(kg(K^-1)(m/s)^2),Densisty(kg/m^3),Potential Energy(kg(m/s)^2),Internal Energy(kg(m/s)^2),Kinetic Energy(kg(m/s)^2),Total Av. Energy(kg(m/s)^2)'+'\n')
        for i in range(102):
            #writefile.write('{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}'.format(fraclistcore[i],Slistcore[i],rholistcore[i],potlistcore[i],ulistcore[i],KElistcore[i],Etotlistcore[i])+'\n')
            writefile.write('{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}'.format(fraclistRubiecore[i],SCMBrubie,rholistRubiecore[i],potlistRubiecore[i],ulistRubiecore[i],KElistRubiecore[i],EtotlistRubiecore[i])+'\n')
    print('Saved file ',basename+'HERCULES_Rubie_core_profiles.csv')
    
    fraclistRubiemant = np.zeros(102)
    #masslistRubiemant = np.zeros(102)
    SlistRubiemant = np.zeros(102)
    rholistRubiemant = np.zeros(102)
    potlistRubiemant = np.zeros(102)
    ulistRubiemant = np.zeros(102)
    KElistRubiemant = np.zeros(102)
    EtotlistRubiemant = np.zeros(102)
    
    for i in range(100):
        fraclistRubiemant[i+1]=0.005+i*0.01
        r1 = (3/4/np.pi/rhomant*(mmantle/100)+r0**3)**(1/3)
        player = P_rubie((r1+r0)/2, 0, 0, (mcore+mmantle)/M_earth, mcore, mmantle)
        rholistRubiemant[i+1] = rhomant
        SlistRubiemant[i+1] = sesame.s_rho_T(float(rhomant),T_rubie(player),int(mant_mat_id))
        potlistRubiemant[i+1] = -G*(mcore+(mmantle/100)*(i+1))/((r1+r0)/2)
        ulistRubiemant[i+1] = sesame.u_rho_T(rhomant,T_rubie(player),mant_mat_id)
        KElistRubiemant[i+1] = 1.
        EtotlistRubiemant[i+1] = potlistRubiemant[i+1]+ulistRubiemant[i+1]
        r0=r1
        
    fraclistRubiemant[0]=0
    fraclistRubiemant[101]=1
    SlistRubiemant[0]=SlistRubiemant[1]
    SlistRubiemant[101]=SlistRubiemant[100]
    rholistRubiemant[0]=rholistRubiemant[1]
    rholistRubiemant[101]=rholistRubiemant[100]
    potlistRubiemant[0]=potlistRubiemant[1]
    potlistRubiemant[101]=potlistRubiemant[100]
    ulistRubiemant[0]=ulistRubiemant[1]
    ulistRubiemant[101]=ulistRubiemant[100]
    KElistRubiemant[0]=KElistRubiemant[1]
    KElistRubiemant[101]=KElistRubiemant[100]
    EtotlistRubiemant[0]=EtotlistRubiemant[1]
    EtotlistRubiemant[101]=EtotlistRubiemant[100]
    
    with open(basename+'HERCULES_Rubie_mant_profiles.csv','w') as writefile:
        writefile.write('0'+'\n'+basename+''+'\n'+'""'+'\n'+'""'+'\n'+'""'+'\n'+str(np.sum(Lz,dtype='float64'))+'\n'+str(mmantle)+'|'+str(mcore)+'\n'+str(r1)+'|'+str(rCMBrubie)+'\n')
        writefile.write('Mass(kg),Entropy(kg(K^-1)(m/s)^2),Densisty(kg/m^3),Potential Energy(kg(m/s)^2),Internal Energy(kg(m/s)^2),Kinetic Energy(kg(m/s)^2),Total Av. Energy(kg(m/s)^2)'+'\n')
        for i in range(102):
            writefile.write('{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e}'.format(fraclistRubiemant[i],SlistRubiemant[i],rholistRubiemant[i],potlistRubiemant[i],ulistRubiemant[i],KElistRubiemant[i],EtotlistRubiemant[i])+'\n')
    print('Saved file ',basename+'HERCULES_Rubie_mant_profiles.csv')
    
    mat_amax_string = '[{:.6e},{:.6e}]'.format(r1,rCMBrubie)
    
    yamldata = dict(
        Main = dict(
            flag_version = 2,
            flag_subversion = 0,
            flag_run_mode = [0]),
        Initialization = dict(
            flag_start = 0,
            start_file = None,
            flag_format_start_file = 1,
            aspect_init = 0.95,
            material_amax_init = [float(r1),float(rCMBrubie)]),
        Output = dict(
            run_name_base = data.metadata.parameters.get('Snapshots:basename').decode(),
            flag_naming = 0,
            flag_output_format = 0,
            output_dir = './',
            flag_iter_print = 0,
            thermo_var_output = ['T', 'S', 'E']),
        Iteration = dict(
            nint_max = 20,
            toll = 1.e-4,
            xi_nint_max = 200,
            xi_toll = 1.e-6,
            dxi = 1.e-3,
            flag_Mconc = 2,
            flag_Lconc = 1,
            omega_param = [1.0,0.0,1.5]),
        Planet = dict(
            Nlayers = 150,
            Nmaterial = 2,
            material_lay = [100,50],
            material_flag_EOS = [2,2],
            EOS_files_std = ['/home/apostema/hercules/HERCULES_development/Source_code/Pyrolite_NEW-GADGET-STD-NOTENSION.TXT','/home/apostema/hercules/HERCULES_development/Source_code/Fe85Si15_NEW-GADGET-STD-NOTENSION.TXT'],
            flag_thermo_profile = [1,1],
            thermo_profile_files = [data.metadata.parameters.get('Snapshots:basename').decode()+'_HERCULES_Rubie_mant_profiles.csv',data.metadata.parameters.get('Snapshots:basename').decode()+'_HERCULES_Rubie_core_profiles.csv'],
            Mass = [float(mmantle),float(mcore)],
            ref_rho = 1.e3,
            kmax = 6,
            Nmu = 400,
            omega_rot = 0.0e-4,
            Ltot = float(np.sum(Lz[index],dtype='float64')),
            pmin = 1.e6),
        Planet_array_calc = dict(
            Xfinal = 10.E34,
            Xstep = 1.0E33,
            Ndiv_Xstep = 1,
            flag_array_toll = 1,
            toll_array = 1E-4,
            xi_toll_array = [1E-6])
        )
    
    with open(basename+'input2.yml', 'w') as outfile:
        yaml.dump(yamldata, outfile, default_flow_style=None, sort_keys=False)

def Lz_profiles(data,cm,vcm,pos,vel,index=None):
    boxsize = data.metadata.boxsize.value
    mat_id = data.gas.material_ids.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth #- cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth #- cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth #- cm[2]
    vel_x = vel.value[:,0] #- vcm[0]
    vel_y = vel.value[:,1] #- vcm[1]
    vel_z = vel.value[:,2] #- vcm[2]
    m = data.gas.masses.value
    Lz = np.abs(m*(pos_x*vel_y - pos_y*vel_x))
    #indcore = np.where(mat_id==core_mat_id)
    #indmant = np.where(mat_id==mant_mat_id)
    
    if index is None:
        index=np.where(m>0)
    
    rr = np.sqrt((pos_x)**2 + (pos_y)**2 + (pos_z)**2)
    RRsort = np.argsort(rr[index])
    indcore = np.where(mat_id[index]==core_mat_id)
    indmant = np.where(mat_id[index]==mant_mat_id)
    
    radiusmeanlist=np.array([])
    Lz50list=np.array([])
    Lz25list=np.array([])
    Lz75list=np.array([])
    Lzmeanlist=np.array([])
    Lzsumlist=np.array([])
    
    for i in range(0,np.size(RRsort),100):
        radiusmean=np.mean(rr[index][RRsort][i:np.amin([np.size(RRsort),i+100])])
        radiusmeanlist = np.append(radiusmeanlist,radiusmean)
        Lz25=np.percentile(Lz[index][RRsort][i:np.amin([np.size(RRsort),i+100])],25)
        Lz25list = np.append(Lz25list,Lz25)
        Lz50=np.percentile(Lz[index][RRsort][i:np.amin([np.size(RRsort),i+100])],50)
        Lz50list = np.append(Lz50list,Lz50)
        Lz75=np.percentile(Lz[index][RRsort][i:np.amin([np.size(RRsort),i+100])],75)
        Lz75list = np.append(Lz75list,Lz75)
        Lzmean=np.mean(Lz[index][RRsort][i:np.amin([np.size(RRsort),i+100])])
        Lzmeanlist = np.append(Lzmeanlist,Lzmean)
        Lzsum=np.sum(Lz[index][RRsort][i:np.amin([np.size(RRsort),i+100])],dtype='float64')
        Lzsumlist = np.append(Lzsumlist,Lzsum)
        
    
    with open(basename+'_L_profiles.txt','w') as writefile:
        writefile.write('#RRR|Lz25|Lz50|Lz75|Lzmean|'+'\n')
        for i in range(np.size(radiusmeanlist)):
            writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(radiusmeanlist[i],Lz25list[i],Lz50list[i],Lz75list[i],Lzmeanlist[i],Lzsumlist[i])+'\n')
        
    corecolor='dimgray'
    mantcolor='indianred'
    fig, ax = plt.subplots(1, 1, figsize=(12,12))
    ax.scatter(pos_x[index][indmant]/R_earth,pos_y[index][indmant]/R_earth,s=3,color=mantcolor,label='Mantle particles')
    ax.scatter(pos_x[index][indcore]/R_earth,pos_y[index][indcore]/R_earth,s=3,color=corecolor,label='Core particles')
    #ax.plot(radiusmeanlist/R_earth,Lz25list)
    #ax.plot(radiusmeanlist/R_earth,Lz50list)
    #ax.plot(radiusmeanlist/R_earth,Lz75list)
    #ax.plot(radiusmeanlist/R_earth,Lzmeanlist)
    ax.set_xlabel(r"X $[R_\oplus]$")
    ax.set_ylabel(r"Y $[R_\oplus]$")
    ax.tick_params(axis='both', which='both', labelsize=18)
    #ax.set_yscale('log')
    
    plt.tight_layout()
    imname=basename+'_all_bound.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    corecolor='dimgray'
    mantcolor='indianred'
    fig, ax = plt.subplots(1, 1, figsize=(12,12))
    ax.scatter(rr[index][indcore]/R_earth,Lz[index][indcore],s=2,color=corecolor,label='Core particles')
    ax.scatter(rr[index][indmant]/R_earth,Lz[index][indmant],s=2,color=mantcolor,label='Mantle particles')
    ax.plot(radiusmeanlist/R_earth,Lz25list)
    ax.plot(radiusmeanlist/R_earth,Lz50list)
    ax.plot(radiusmeanlist/R_earth,Lz75list)
    ax.plot(radiusmeanlist/R_earth,Lzmeanlist)
    ax.set_xlabel(r"R $[R_\oplus]$")
    ax.set_ylabel(r'Lz $')
    ax.tick_params(axis='both', which='both', labelsize=18)
    #ax.set_yscale('log')
    
    plt.tight_layout()
    imname=basename+'_Lz.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    

def isentrope_profiles(data,cm,vcm,pos,vel,rplanet=None,index=None,basename=''):
    if basename != '':
        basename = basename+'_'
    u = data.gas.internal_energies.value
    P = data.gas.pressures.value
    m = data.gas.masses.value
    mtot = np.sum(m,dtype='float64')
    utot = np.sum(u*m,dtype='float64')
    rho = data.gas.densities.value
    mat_id = data.gas.material_ids.value
    part_id = data.gas.progenitor_particle_ids.value
    tmp_s = np.zeros(np.size(u))
    tmp_T = np.zeros(np.size(u))
    for i in range(np.size(u)):
        tmp_s[i] = sesame.s_u_rho(u[i],rho[i],mat_id[i])
        tmp_T[i] = sesame.T_rho_s(rho[i],tmp_s[i],mat_id[i])
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth #- cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth #- cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth #- cm[2]
    #snaptime = data.metadata.time.in_mks()
    
    rrr = np.sqrt((pos_x)**2 + (pos_y)**2 + (pos_z)**2)
    
    if index is None:
        index=np.where(m>0)
    
    if rplanet is None:
        zlim = np.std(pos_z[index])/3
        rplanet=zlim*10/R_earth
    else:
        zlim = rplanet*R_earth/10
    #indEq = np.where(np.abs(pos_z[index])<(zlim))
    #indEqPlanet = np.where((np.abs(pos_z[index])<(zlim)) & (rrr[index]<=rplanet*R_earth))
    #indmantEq = np.where((np.abs(pos_z[index])<(zlim)) & (mat_id[index]==mant_mat_id))
    indmantEqPlanet = np.where((np.abs(pos_z[index])<(zlim)) & (rrr[index]<=rplanet*R_earth) & (mat_id[index]==mant_mat_id))
    #indcoreEq = np.where((np.abs(pos_z[index])<(zlim)) & (mat_id[index]==core_mat_id))
    indcoreEqPlanet = np.where((np.abs(pos_z[index])<(zlim)) & (rrr[index]<=rplanet*R_earth) & (mat_id[index]==core_mat_id))
    
    utotCore = np.sum(u[index][indcoreEqPlanet]*m[index][indcoreEqPlanet],dtype='float64')
    utotMant = np.sum(u[index][indmantEqPlanet]*m[index][indmantEqPlanet],dtype='float64')
    
    Scorestart = np.mean(tmp_s[index][indcoreEqPlanet])
    Smantstart = np.mean(tmp_s[index][indmantEqPlanet])
    Scoremin = np.amin(tmp_s[index][indcoreEqPlanet])-500
    Scoremax = np.amax(tmp_s[index][indcoreEqPlanet])+500
    Smantmin = np.amin(tmp_s[index][indmantEqPlanet])-500
    Smantmax = np.amax(tmp_s[index][indmantEqPlanet])+500
    
    # Starting loops to iterate to find equivalent-energy isentrope from rho-S
    n=0
    tolerance = 0.001
    Scorenew = Scorestart
    Smantnew = Smantstart
    tmp_core_u = np.zeros(np.size(indcoreEqPlanet))
    tmp_mant_u = np.zeros(np.size(indmantEqPlanet))
    for i in range(np.size(indcoreEqPlanet)):
        tmp_core_u[i] = sesame.u_s_rho(Scorestart,rho[index][indcoreEqPlanet][i],mat_id[index][indcoreEqPlanet][i])
    for i in range(np.size(indmantEqPlanet)):
        tmp_mant_u[i] = sesame.u_s_rho(Smantstart,rho[index][indmantEqPlanet][i],mat_id[index][indmantEqPlanet][i])
        
    utotCoreNew = np.sum(tmp_core_u*m[index][indcoreEqPlanet],dtype='float64')
    utotMantNew = np.sum(tmp_mant_u*m[index][indmantEqPlanet],dtype='float64')
    
    while (np.abs(utotCore-utotCoreNew)/utotCore > tolerance) & (n<50):
        if (utotCore-utotCoreNew) > 0:
            Scorenew = (Scorestart+Scoremax)/2
            Scoremin = Scorestart
        if (utotCore-utotCoreNew) < 0:
            Scorenew = (Scorestart+Scoremin)/2
            Scoremax = Scorestart
        for i in range(np.size(indcoreEqPlanet)):
            tmp_core_u[i] = sesame.u_s_rho(Scorenew,rho[index][indcoreEqPlanet][i],mat_id[index][indcoreEqPlanet][i])
        utotCoreNew = np.sum(tmp_core_u*m[index][indcoreEqPlanet],dtype='float64')
        print('OLD SCore: ',Scorestart/1.e3,'kJ/kg/K NEW SCore: ',Scorenew/1.e3,'kJ/kg/K')
        print('CORE ENERGY DIFF: ',(utotCoreNew-utotCore)/utotCore,' TOLERANCE: ',tolerance)
        Scorestart=Scorenew
        n+=1
        
    n=0
    while (np.abs(utotMant-utotMantNew)/utotMant > tolerance) & (n<50):
        if (utotMant-utotMantNew) > 0:
            Smantnew = (Smantstart+Smantmax)/2
            Smantmin = Smantstart
        if (utotMant-utotMantNew) < 0:
            Smantnew = (Smantstart+Smantmin)/2
            Smantmax = Smantstart
        for i in range(np.size(indmantEqPlanet)):
            tmp_mant_u[i] = sesame.u_s_rho(Smantnew,rho[index][indmantEqPlanet][i],mat_id[index][indmantEqPlanet][i])
        utotMantNew = np.sum(tmp_mant_u*m[index][indmantEqPlanet],dtype='float64')
        print('OLD SMant: ',Smantstart/1.e3,'kJ/kg/K NEW SMant: ',Smantnew/1.e3,'kJ/kg/K')
        print('MANTLE ENERGY DIFF: ',(utotMantNew-utotMant)/utotMant,' TOLERANCE: ',tolerance)
        Smantstart=Smantnew
        n+=1
        
    tmp_core_T = np.zeros(np.size(indcoreEqPlanet))
    tmp_core_P = np.zeros(np.size(indcoreEqPlanet))
    tmp_mant_T = np.zeros(np.size(indmantEqPlanet))
    tmp_mant_P = np.zeros(np.size(indmantEqPlanet))
    for i in range(np.size(indcoreEqPlanet)):
        tmp_core_T[i] = sesame.T_rho_s(rho[index][indcoreEqPlanet][i],Scorenew,mat_id[index][indcoreEqPlanet][i])
        tmp_core_P[i] = sesame.P_u_rho(tmp_core_u[i],rho[index][indcoreEqPlanet][i],mat_id[index][indcoreEqPlanet][i])
    for i in range(np.size(indmantEqPlanet)):
        tmp_mant_T[i] = sesame.T_rho_s(rho[index][indmantEqPlanet][i],Smantnew,mat_id[index][indmantEqPlanet][i])
        tmp_mant_P[i] = sesame.P_u_rho(tmp_mant_u[i],rho[index][indmantEqPlanet][i],mat_id[index][indmantEqPlanet][i])
    
    Rsortcore = np.argsort(rrr[index][indcoreEqPlanet])
    Rsortmant = np.argsort(rrr[index][indmantEqPlanet])    
    radiusmeancorelist=np.array([])
    radiusmeanmantlist=np.array([])
    radiusEqcore = rrr[index][indcoreEqPlanet]
    radiusEqmant = rrr[index][indmantEqPlanet]
    rhoEqcore = rho[index][indcoreEqPlanet]
    rhoEqmant = rho[index][indmantEqPlanet]
    
    u50corelist=np.array([])
    u25corelist=np.array([])
    u75corelist=np.array([])
    umeancorelist=np.array([])
    
    u50mantlist=np.array([])
    u25mantlist=np.array([])
    u75mantlist=np.array([])
    umeanmantlist=np.array([])
    
    T50corelist=np.array([])
    T25corelist=np.array([])
    T75corelist=np.array([])
    Tmeancorelist=np.array([])
    
    T50mantlist=np.array([])
    T25mantlist=np.array([])
    T75mantlist=np.array([])
    Tmeanmantlist=np.array([])
    
    P50corelist=np.array([])
    P25corelist=np.array([])
    P75corelist=np.array([])
    Pmeancorelist=np.array([])
    
    P50mantlist=np.array([])
    P25mantlist=np.array([])
    P75mantlist=np.array([])
    Pmeanmantlist=np.array([])
    
    rho50corelist=np.array([])
    rho25corelist=np.array([])
    rho75corelist=np.array([])
    rhomeancorelist=np.array([])
    
    rho50mantlist=np.array([])
    rho25mantlist=np.array([])
    rho75mantlist=np.array([])
    rhomeanmantlist=np.array([])
    
    with open(basename+'isentropes.txt','w') as writefile:
        writefile.write('#RRR|U50|U25|U75|Umean|T50|T25|T75|Tmean|P50|P25|P75|Pmean|rho50|rho25|rho75|Pmean|S'+'\n')
        for i in range(0,np.size(Rsortcore),100):
            radiusmean=np.mean(radiusEqcore[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])])
            radiusmeancorelist=np.append(radiusmeancorelist,radiusmean)    
            
            u25=np.percentile(tmp_core_u[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],25)
            u25corelist=np.append(u25corelist,u25)
            u50=np.percentile(tmp_core_u[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],50)
            u50corelist=np.append(u50corelist,u50)
            u75=np.percentile(tmp_core_u[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],75)
            u75corelist=np.append(u75corelist,u75)
            umean=np.mean(tmp_core_u[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])])
            umeancorelist=np.append(umeancorelist,umean)
            
            T25=np.percentile(tmp_core_T[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],25)
            T25corelist=np.append(T25corelist,T25)
            T50=np.percentile(tmp_core_T[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],50)
            T50corelist=np.append(T50corelist,T50)
            T75=np.percentile(tmp_core_T[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],75)
            T75corelist=np.append(T75corelist,T75)
            Tmean=np.mean(tmp_core_T[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])])
            Tmeancorelist=np.append(Tmeancorelist,Tmean)
            
            P25=np.percentile(tmp_core_P[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],25)
            P25corelist=np.append(P25corelist,P25)
            P50=np.percentile(tmp_core_P[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],50)
            P50corelist=np.append(P50corelist,P50)
            P75=np.percentile(tmp_core_P[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],75)
            P75corelist=np.append(P75corelist,P75)
            Pmean=np.mean(tmp_core_P[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])])
            Pmeancorelist=np.append(Pmeancorelist,Pmean)
            
            rho25=np.percentile(rhoEqcore[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],25)
            rho25corelist=np.append(rho25corelist,rho25)
            rho50=np.percentile(rhoEqcore[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],50)
            rho50corelist=np.append(rho50corelist,rho50)
            rho75=np.percentile(rhoEqcore[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])],75)
            rho75corelist=np.append(rho75corelist,rho75)
            rhomean=np.mean(rhoEqcore[Rsortcore][i:np.amin([np.size(Rsortcore),i+100])])
            rhomeancorelist=np.append(rhomeancorelist,rhomean)
            
            writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(radiusmean,u25,u50,u75,umean,T25,T50,T75,Tmean,P25,P50,P75,Pmean,rho25,rho50,rho75,rhomean,Scorenew)+'\n')
            
        for i in range(0,np.size(Rsortmant),100):
            radiusmean=np.mean(radiusEqmant[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])])
            radiusmeanmantlist=np.append(radiusmeanmantlist,radiusmean)    
            
            u25=np.percentile(tmp_mant_u[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],25)
            u25mantlist=np.append(u25mantlist,u25)
            u50=np.percentile(tmp_mant_u[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],50)
            u50mantlist=np.append(u50mantlist,u50)
            u75=np.percentile(tmp_mant_u[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],75)
            u75mantlist=np.append(u75mantlist,u75)
            umean=np.mean(tmp_mant_u[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])])
            umeanmantlist=np.append(umeanmantlist,umean)
            
            T25=np.percentile(tmp_mant_T[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],25)
            T25mantlist=np.append(T25mantlist,T25)
            T50=np.percentile(tmp_mant_T[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],50)
            T50mantlist=np.append(T50mantlist,T50)
            T75=np.percentile(tmp_mant_T[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],75)
            T75mantlist=np.append(T75mantlist,T75)
            Tmean=np.mean(tmp_mant_T[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])])
            Tmeanmantlist=np.append(Tmeanmantlist,Tmean)
            
            P25=np.percentile(tmp_mant_P[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],25)
            P25mantlist=np.append(P25mantlist,P25)
            P50=np.percentile(tmp_mant_P[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],50)
            P50mantlist=np.append(P50mantlist,P50)
            P75=np.percentile(tmp_mant_P[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],75)
            P75mantlist=np.append(P75mantlist,P75)
            Pmean=np.mean(tmp_mant_P[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])])
            Pmeanmantlist=np.append(Pmeanmantlist,Pmean)
            
            rho25=np.percentile(rhoEqmant[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],25)
            rho25mantlist=np.append(rho25mantlist,rho25)
            rho50=np.percentile(rhoEqmant[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],50)
            rho50mantlist=np.append(rho50mantlist,rho50)
            rho75=np.percentile(rhoEqmant[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])],75)
            rho75mantlist=np.append(rho75mantlist,rho75)
            rhomean=np.mean(rhoEqmant[Rsortmant][i:np.amin([np.size(Rsortmant),i+100])])
            rhomeanmantlist=np.append(rhomeanmantlist,rhomean)
            
            writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(radiusmean,u25,u50,u75,umean,T25,T50,T75,Tmean,P25,P50,P75,Pmean,rho25,rho50,rho75,rhomean,Smantnew)+'\n')
    print('Saved file ',basename+'isentropes.txt')
    
    plotlimitcore = np.where(P25corelist>1.e6)
    plotlimitmant = np.where(P25mantlist>1.e6)
    
    corecolor='dimgray'
    mantcolor='indianred'
    fig, ax = plt.subplots(2, 2, figsize=(18,18))
    ax[0,0].scatter(radiusEqcore/R_earth,rhoEqcore,s=2,color=corecolor,label='Core particles')
    ax[0,0].scatter(radiusEqmant/R_earth,rhoEqmant,s=2,color=mantcolor,label='Core particles')
    ax[0,0].plot(radiusmeancorelist/R_earth,rho25corelist)
    ax[0,0].plot(radiusmeancorelist/R_earth,rho50corelist)
    ax[0,0].plot(radiusmeancorelist/R_earth,rho75corelist)
    ax[0,0].plot(radiusmeancorelist/R_earth,rhomeancorelist)
    ax[0,0].plot(radiusmeanmantlist/R_earth,rho25mantlist)
    ax[0,0].plot(radiusmeanmantlist/R_earth,rho50mantlist)
    ax[0,0].plot(radiusmeanmantlist/R_earth,rho75mantlist)
    ax[0,0].plot(radiusmeanmantlist/R_earth,rhomeanmantlist)
    ax[0,0].set_xlabel(r"R $[R_\oplus]$")
    ax[0,0].set_ylabel(r'Density $rho [g/cm^3]$')
    ax[0,0].tick_params(axis='both', which='both', labelsize=18)
    ax[0,0].set_yscale('log')
    
    ax[0,1].scatter(radiusEqcore/R_earth,tmp_core_P/1.e9,s=2,color=corecolor,label='Core particles')
    ax[0,1].scatter(radiusEqmant/R_earth,tmp_mant_P/1.e9,s=2,color=mantcolor,label='Core particles')
    ax[0,1].plot(radiusmeancorelist[plotlimitcore]/R_earth,P25corelist[plotlimitcore]/1.e9)
    ax[0,1].plot(radiusmeancorelist[plotlimitcore]/R_earth,P50corelist[plotlimitcore]/1.e9)
    ax[0,1].plot(radiusmeancorelist[plotlimitcore]/R_earth,P75corelist[plotlimitcore]/1.e9)
    ax[0,1].plot(radiusmeancorelist[plotlimitcore]/R_earth,Pmeancorelist[plotlimitcore]/1.e9)
    ax[0,1].plot(radiusmeanmantlist[plotlimitmant]/R_earth,P25mantlist[plotlimitmant]/1.e9)
    ax[0,1].plot(radiusmeanmantlist[plotlimitmant]/R_earth,P50mantlist[plotlimitmant]/1.e9)
    ax[0,1].plot(radiusmeanmantlist[plotlimitmant]/R_earth,P75mantlist[plotlimitmant]/1.e9)
    ax[0,1].plot(radiusmeanmantlist[plotlimitmant]/R_earth,Pmeanmantlist[plotlimitmant]/1.e9)
    ax[0,1].set_xlabel(r"R $[R_\oplus]$")
    ax[0,1].set_ylabel(r'Isentrope Pressure P [GPa]')
    ax[0,1].tick_params(axis='both', which='both', labelsize=18)
    
    ax[1,0].scatter(radiusEqcore/R_earth,tmp_core_u/1.e6,s=2,color=corecolor,label='Core particles')
    ax[1,0].scatter(radiusEqmant/R_earth,tmp_mant_u/1.e6,s=2,color=mantcolor,label='Core particles')
    ax[1,0].plot(radiusmeancorelist[plotlimitcore]/R_earth,u25corelist[plotlimitcore]/1.e6)
    ax[1,0].plot(radiusmeancorelist[plotlimitcore]/R_earth,u50corelist[plotlimitcore]/1.e6)
    ax[1,0].plot(radiusmeancorelist[plotlimitcore]/R_earth,u75corelist[plotlimitcore]/1.e6)
    ax[1,0].plot(radiusmeancorelist[plotlimitcore]/R_earth,umeancorelist[plotlimitcore]/1.e6)
    ax[1,0].plot(radiusmeanmantlist[plotlimitmant]/R_earth,u25mantlist[plotlimitmant]/1.e6)
    ax[1,0].plot(radiusmeanmantlist[plotlimitmant]/R_earth,u50mantlist[plotlimitmant]/1.e6)
    ax[1,0].plot(radiusmeanmantlist[plotlimitmant]/R_earth,u75mantlist[plotlimitmant]/1.e6)
    ax[1,0].plot(radiusmeanmantlist[plotlimitmant]/R_earth,umeanmantlist[plotlimitmant]/1.e6)
    ax[1,0].set_xlabel(r"R $[R_\oplus]$")
    ax[1,0].set_ylabel(r'Isentrope Sp. Internal Energy U [MJ/kg]')
    ax[1,0].tick_params(axis='both', which='both', labelsize=18)
    
    ax[1,1].scatter(radiusEqcore/R_earth,tmp_core_T,s=2,color=corecolor,label='Core particles')
    ax[1,1].scatter(radiusEqmant/R_earth,tmp_mant_T,s=2,color=mantcolor,label='Core particles')
    ax[1,1].plot(radiusmeancorelist[plotlimitcore]/R_earth,T25corelist[plotlimitcore])
    ax[1,1].plot(radiusmeancorelist[plotlimitcore]/R_earth,T50corelist[plotlimitcore])
    ax[1,1].plot(radiusmeancorelist[plotlimitcore]/R_earth,T75corelist[plotlimitcore])
    ax[1,1].plot(radiusmeancorelist[plotlimitcore]/R_earth,Tmeancorelist[plotlimitcore])
    ax[1,1].plot(radiusmeanmantlist[plotlimitmant]/R_earth,T25mantlist[plotlimitmant])
    ax[1,1].plot(radiusmeanmantlist[plotlimitmant]/R_earth,T50mantlist[plotlimitmant])
    ax[1,1].plot(radiusmeanmantlist[plotlimitmant]/R_earth,T75mantlist[plotlimitmant])
    ax[1,1].plot(radiusmeanmantlist[plotlimitmant]/R_earth,Tmeanmantlist[plotlimitmant])
    ax[1,1].set_xlabel(r"R $[R_\oplus]$")
    ax[1,1].set_ylabel(r'Isentrope Temperature T [K]')
    ax[1,1].tick_params(axis='both', which='both', labelsize=18)
    
    plt.tight_layout()
    imname=basename+'isentrope_profiles.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
def plot_miscibility(data,cm,vcm,pos,vel,index=None,rplanet=None,basename=''):
    if basename != '':
        basename = basename+'_'
    u = data.gas.internal_energies.value
    P = data.gas.pressures.value
    m = data.gas.masses.value
    mtot = np.sum(m[index],dtype='float64')
    utot = np.sum(u*m,dtype='float64')
    rho = data.gas.densities.value
    mat_id = data.gas.material_ids.value
    part_id = data.gas.progenitor_particle_ids.value
    tmp_s = np.zeros(np.size(u))
    tmp_T = np.zeros(np.size(u))
    for i in range(np.size(u)):
        tmp_s[i] = sesame.s_u_rho(u[i],rho[i],mat_id[i])
        tmp_T[i] = sesame.T_rho_s(rho[i],tmp_s[i],mat_id[i])
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth #- cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth #- cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth #- cm[2]
   
    #snaptime = data.metadata.time.in_mks()   
    rrr = np.sqrt((pos_x)**2 + (pos_y)**2 + (pos_z)**2)
    
    solvusfn = interpolate.interp1d([0,50.e9,100.e9,400.e9,1000.e9],[4080,6094,6752,9337,14507.0])
    
    if index is None:
        index=np.where(m>0)
    
    if rplanet is None:
        zlim = np.std(pos_z[index])/3
        rplanet=zlim*10/R_earth
    else:
        zlim = rplanet*R_earth/10
    #indEq = np.where(np.abs(pos_z[index])<(zlim))
    #indEqPlanet = np.where((np.abs(pos_z[index])<(zlim)) & (rrr[index]<=rplanet*R_earth))
    indmant = np.where(mat_id[index]==mant_mat_id)
    indmantEq = np.where((np.abs(pos_z[index])<(zlim)) & (mat_id[index]==mant_mat_id))
    #indmantEqPlanet = np.where((np.abs(pos_z[index])<(zlim)) & (rrr[index]<=rplanet*R_earth) & (mat_id[index]==mant_mat_id))
    indcore = np.where(mat_id[index]==core_mat_id)
    indcoreEq = np.where((np.abs(pos_z[index])<(zlim)) & (mat_id[index]==core_mat_id))
    #indcoreEqPlanet = np.where((np.abs(pos_z[index])<(zlim)) & (rrr[index]<=rplanet*R_earth) & (mat_id[index]==core_mat_id))
    
    
    coredeltaTsolvus = tmp_T[index][indcoreEq]-solvusfn(P[index][indcoreEq])
    print('Min/max coreDeltaTSolvus: ',np.amin(coredeltaTsolvus), np.amax(coredeltaTsolvus))
    mantdeltaTsolvus = tmp_T[index][indmantEq]-solvusfn(P[index][indmantEq])
    print('Min/max mantDeltaTSolvus: ',np.amin(mantdeltaTsolvus), np.amax(mantdeltaTsolvus))
    mantnorm = TwoSlopeNorm(vmin=np.amin([np.amin(mantdeltaTsolvus),-1.e-4]),vcenter=0,vmax=np.amax([np.amax(mantdeltaTsolvus),1.e-4]))
    mantle=plt.scatter(pos_x[index][indmantEq]/R_earth,pos_y[index][indmantEq]/R_earth,c=(mantdeltaTsolvus),cmap='bwr',norm=mantnorm,s=3)
    corenorm = TwoSlopeNorm(vmin=np.amin([np.amin(coredeltaTsolvus),-1.e-4]),vcenter=0,vmax=np.amax([np.amax(coredeltaTsolvus),1.e-4]))
    core=plt.scatter(pos_x[index][indcoreEq]/R_earth,pos_y[index][indcoreEq]/R_earth,c=(coredeltaTsolvus),cmap='RdGy_r',norm=corenorm,s=3)
    cmant=plt.colorbar(mantle,cmap='bwr',norm=mantnorm)
    ccore=plt.colorbar(core,cmap='RdGy_r',norm=corenorm)
    cmant.set_label('Mantle T above solvus [K]',size=20)
    #cmant.set_label('Mantle T above solvus, avg = %g K' % (mantSavg/1.e3),size=20)
    #cmant.ax.hlines(mantSavg/1.e3,np.amin([cmant.vmin,0]),cmant.vmax,color='white')
    ccore.set_label('Core T above solvus [K]',size=20)
    #ccore.set_label('Core sp. Entropy gain (kJ/kg/K), avg = %g kJ/kg/K' % (coreSavg/1.e3),size=20)
    #ccore.ax.hlines(coreSavg/1.e3,np.amin([ccore.vmin,0]),ccore.vmax,color='white')
    plt.gca().set_xlabel(r"X $[R_\oplus]$",size=20)
    plt.gca().set_ylabel(r"Y $[R_\oplus]$",size=20)
    plt.gca().set_xlim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_ylim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_aspect('equal')
    plt.gca().set_title('Temperature relative to Fe-MgO solvus closure'+'\n'+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Bound particles: %i"%(np.shape(index)[1]),size=20)
    plt.gcf().set_size_inches(18,12)
    plt.gca().set_facecolor('0.075')
    plt.tight_layout()
    imname = basename+'solvusT.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    mantMisc = np.where((tmp_T[index][indmant]-solvusfn(P[index][indmant]))>=0)
    coreMisc = np.where((tmp_T[index][indcore]-solvusfn(P[index][indcore]))>=0)
    MmantMisc = np.sum(m[index][mantMisc],dtype='float64')
    McoreMisc = np.sum(m[index][coreMisc],dtype='float64')
    mcore = np.sum(m[index][indcore],dtype='float64')
    mmant = np.sum(m[index][indmant],dtype='float64')
    
    with open(basename+'miscibility.txt','w') as writefile:
        writefile.write('#Mtot|MmantMisc|McoreMisc|MmantMisc/Mmant|McoreMisc/Mcore'+'\n')
        writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(mtot/M_earth,MmantMisc/M_earth,McoreMisc/M_earth,MmantMisc/mmant,McoreMisc/mcore)+'\n')
    print('Saved file ',basename+'miscibility.txt')
    

def plot_IE_gain(data0,data,cm,vcm,pos,vel,index=None,U0=None,rplanet=None,basename=''):
    '''
    Plots internal energy gain of bound mass
    Parameters
    ----------
    data0 : SWIFT dataset from simulation time zero - initial condition
    data : SWIFT dataset from final timestep
    pos : TYPE
        DESCRIPTION.
    vel : TYPE
        DESCRIPTION.
    index : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''
    if basename != '':
        basename = basename+'_'
    u0 = data0.gas.internal_energies.value
    rho0 = data0.gas.densities.value
    mat_id0 = data0.gas.material_ids.value
    P0 = data0.gas.pressures.value
    part_id0 = data0.gas.progenitor_particle_ids.value
    tmp_s0 = np.zeros(np.size(u0))
    tmp_T0 = np.zeros(np.size(u0))
    for i in range(np.size(u0)):
        tmp_s0[i] = sesame.s_u_rho(u0[i],rho0[i],mat_id0[i])
        tmp_T0[i] = sesame.T_rho_s(rho0[i],tmp_s0[i],mat_id0[i])
    u = data.gas.internal_energies.value
    P = data.gas.pressures.value
    m = data.gas.masses.value
    mtot = np.sum(m,dtype='float64')
    rho = data.gas.densities.value
    mat_id = data.gas.material_ids.value
    part_id = data.gas.progenitor_particle_ids.value
    tmp_s = np.zeros(np.size(u))
    tmp_T = np.zeros(np.size(u))
    for i in range(np.size(u)):
        tmp_s[i] = sesame.s_u_rho(u[i],rho[i],mat_id[i])
        tmp_T[i] = sesame.T_rho_s(rho[i],tmp_s[i],mat_id[i])
    
    indexin = np.where(np.isin(part_id0,part_id[index]))
    print(np.where(np.isin(part_id,part_id0,invert=True)))
    datapartsort = np.argsort(part_id)
    dataindexsort = np.argsort(part_id[index])
    data0inpartsort = np.argsort(part_id0[indexin])
    u0inpartsort = u0[indexin][data0inpartsort]
    upartsort = u[datapartsort]
    uindex = u[index]
    uindexsort =u[index][dataindexsort]
    mindex = m[index]
    mindexsort =m[index][dataindexsort]
    matidsort = mat_id[datapartsort]
    matidindexsort=mat_id[index][dataindexsort]
    matid0inpartsort=mat_id0[indexin][data0inpartsort]
    
    #pos=data.gas.coordinates
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth #- cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth #- cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth #- cm[2]
    
    vel_x = vel.value[:,0] #- vcm[0]
    vel_y = vel.value[:,1] #- vcm[1]
    vel_z = vel.value[:,2] #- vcm[2]
    snaptime = data.metadata.time.in_mks()
    
    
    if index is None:
        index = np.where(np.isin(part_id0,part_id))
    #u0in = u0[indexin]
    
    #indmant = np.where(mat_id[index][dataindexsort]==mant_mat_id)
    indmant = np.where(matidindexsort==mant_mat_id)
    mmant = np.sum(m[indmant],dtype='float64')
    indmant0 = np.where(matid0inpartsort==mant_mat_id)
    indcore = np.where(mat_id[index][dataindexsort]==core_mat_id)
    mcore = np.sum(m[indcore],dtype='float64')
    print(np.size(uindexsort),np.size(u0inpartsort))
    print(np.size(part_id0),np.size(part_id[index]),np.size(indexin))
    #mantIEavg = np.sum((uindexsort[indmant]-u0inpartsort[indmant])*(mindexsort[indmant]))/np.sum(mindexsort[indmant])
    mantIEavg = np.sum((u[index][dataindexsort][indmant]-u0[indexin][data0inpartsort][indmant])*(m[index][dataindexsort][indmant]))/np.sum(m[index][dataindexsort][indmant])
    coreIEavg = np.sum((u[index][dataindexsort][indcore]-u0[indexin][data0inpartsort][indcore])*m[index][dataindexsort][indcore])/np.sum(m[index][dataindexsort][indcore])
    print('Average mantle IE gain: ',mantIEavg/1.e6,'[MJ/kg] ',)
    #print('Average mantle particle IE gain: ',np.sum((u[index][dataindexsort][indmant]-u0[indexin][data0inpartsort][indmant]))/np.size(u[index][dataindexsort][indmant])/1.e6,'[MJ/kg per particle] ',)
    print('Average core IE gain: ',coreIEavg/1.e6,'[MJ/kg]')
    #print('Average core particle IE gain: ',np.sum((u[index][dataindexsort][indcore]-u0[indexin][data0inpartsort][indcore]))/np.size(u[index][dataindexsort][indcore])/1.e6,'[MJ/kg per particle] ',)
    mantSavg = np.sum((tmp_s[index][dataindexsort][indmant]-tmp_s0[indexin][data0inpartsort][indmant])*m[index][dataindexsort][indmant])/np.sum(m[index][dataindexsort][indmant])
    coreSavg = np.sum((tmp_s[index][dataindexsort][indcore]-tmp_s0[indexin][data0inpartsort][indcore])*m[index][dataindexsort][indcore])/np.sum(m[index][dataindexsort][indcore])
    print('Average mantle S gain: ',mantSavg/1.e3,'[kJ/kg/K]')
    #print('Average mantle particle S gain: ',np.sum((tmp_s[index][dataindexsort][indmant]-tmp_s0[indexin][data0inpartsort][indmant]))/np.size(tmp_s[index][dataindexsort][indmant])/1.e3,'[kJ/kg/K per particle] ',)
    print('Average core S gain: ',coreSavg/1.e3,'[kJ/kg/K]')
    #print('Average core particle S gain: ',np.sum((tmp_s[index][dataindexsort][indcore]-tmp_s0[indexin][data0inpartsort][indcore]))/np.size(tmp_s[index][dataindexsort][indcore])/1.e3,'[kJ/kg/K per particle] ',)
    mantTavg = np.sum((tmp_T[index][dataindexsort][indmant]-tmp_T0[indexin][data0inpartsort][indmant])*m[index][dataindexsort][indmant])/np.sum(m[index][dataindexsort][indmant])
    coreTavg = np.sum((tmp_T[index][dataindexsort][indcore]-tmp_T0[indexin][data0inpartsort][indcore])*m[index][dataindexsort][indcore])/np.sum(m[index][dataindexsort][indcore])
    print('Average mantle T gain: ',mantTavg,'[K]')
    #print('Average mantle particle T gain: ',np.sum((tmp_T[index][dataindexsort][indmant]-tmp_T0[indexin][data0inpartsort][indmant]))/np.size(tmp_T[index][dataindexsort][indmant]),'[K per particle] ',)
    print('Average core T gain: ',coreTavg,'[K]')
    #print('Average core particle T gain: ',np.sum((tmp_T[index][dataindexsort][indcore]-tmp_T0[indexin][data0inpartsort][indcore]))/np.size(tmp_T[index][dataindexsort][indcore]),'[K per particle]')
    
    rrr = np.sqrt((pos_x)**2 + (pos_y)**2 + (pos_z)**2)
    rr = np.sqrt((pos_x)**2 + (pos_y)**2)
    if rplanet is None:
        zlim = np.std(pos_z[index][dataindexsort])/3
        rplanet=zlim*10/R_earth
    else:
        zlim = rplanet*R_earth/10
    indEq = np.where(np.abs(pos_z[index][dataindexsort])<(zlim))
    indEqPlanet = np.where((np.abs(pos_z[index][dataindexsort])<(zlim)) & (rrr[index][dataindexsort]<=rplanet*R_earth))
    indmantEq = np.where((np.abs(pos_z[index][dataindexsort])<(zlim)) & (mat_id[index][dataindexsort]==mant_mat_id))
    indmantEqPlanet = np.where((np.abs(pos_z[index][dataindexsort])<(zlim)) & (rrr[index][dataindexsort]<=rplanet*R_earth) & (mat_id[index][dataindexsort]==mant_mat_id))
    indcoreEq = np.where((np.abs(pos_z[index][dataindexsort])<(zlim)) & (mat_id[index][dataindexsort]==core_mat_id))
    indcoreEqPlanet = np.where((np.abs(pos_z[index][dataindexsort])<(zlim)) & (rrr[index][dataindexsort]<=rplanet*R_earth) & (mat_id[index][dataindexsort]==core_mat_id))
    
    #plt.scatter(pos_x[index][dataindexsort]/R_earth,pos_y[index][dataindexsort]/R_earth,c=(u[index][dataindexsort]-u0[indexin][data0inpartsort]),s=1)
    mantle=plt.scatter(pos_x[index][dataindexsort][indmantEq]/R_earth,pos_y[index][dataindexsort][indmantEq]/R_earth,c=(u[index][dataindexsort][indmantEq]-u0[indexin][data0inpartsort][indmantEq])/1.e6,cmap='inferno',s=3)
    core=plt.scatter(pos_x[index][dataindexsort][indcoreEq]/R_earth,pos_y[index][dataindexsort][indcoreEq]/R_earth,c=(u[index][dataindexsort][indcoreEq]-u0[indexin][data0inpartsort][indcoreEq])/1.e6,cmap='pink',s=3)
    #plt.scatter(pos_x[indcoreEq]/R_earth,pos_y[indcoreEq]/R_earth,c=(u[indcoreEq]-u0[indcoreEq]),s=1)
    cmant=plt.colorbar(mantle,cmap='inferno')
    ccore=plt.colorbar(core,cmap='pink')
    cmant.set_label('Mantle IE gain (MJ/kg), avg = %g MJ/kg' % (mantIEavg/1.e6),size=20)
    cmant.ax.hlines(mantIEavg/1.e6,np.amin([cmant.vmin,0]),cmant.vmax,color='white')
    ccore.set_label('Core IE gain (MJ/kg), avg = %g MJ/kg' % (coreIEavg/1.e6),size=20)
    ccore.ax.hlines(coreIEavg/1.e6,np.amin([ccore.vmin,0]),ccore.vmax,color='white')
    plt.gca().set_xlabel(r"X $[R_\oplus]$",size=20)
    plt.gca().set_ylabel(r"Y $[R_\oplus]$",size=20)
    plt.gca().set_xlim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_ylim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_aspect('equal')
    plt.gca().set_title('Sp. Internal Energy Gain'+'\n'+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Bound particles: %i"%(np.shape(index)[1]),size=20)
    plt.gcf().set_size_inches(18,12)
    plt.gca().set_facecolor('0.075')
    # fir, ax = plt.subplots(1, 1, figsize=(10,6))
    # ax.scatter(pos_x[indmantEq]/R_earth,pos_y[indmantEq]/R_earth,c=(u[indmantEq]-u0[indmantEq]),s=1)
    # ax.scatter(pos_x[indcoreEq]/R_earth,pos_y[indcoreEq]/R_earth,c=(u[indcoreEq]-u0[indcoreEq]),s=1)
    # #fir.colorbar((u[indmantEq]-u0[indmantEq]),cax=ax,cmap='inferno')
    # fir.colorbar(,cax=ax,cmap='inferno')
    # #fir.colorbar((u[indcoreEq]-u0[indcoreEq]),cax=ax,cmap='gray')
    # ax.set_xlabel(r"X $[R_\oplus]$")
    # ax.set_ylabel(r"Y $[R_\oplus]$")
    # ax.axis('equal')
    # ax.set_xlim(-5*z_std/R_earth, 5*z_std/R_earth)
    # ax.set_ylim(-5*z_std/R_earth, 5*z_std/R_earth)
    # ax.set_title("Time %06d s" % (snaptime))
    
    plt.tight_layout()
    imname = basename+'IE_gain.png'
    plt.savefig(imname, dpi=100)
    plt.close()

    # plotting specific entropy gain
    mantle=plt.scatter(pos_x[index][dataindexsort][indmantEq]/R_earth,pos_y[index][dataindexsort][indmantEq]/R_earth,c=(tmp_s[index][dataindexsort][indmantEq]-tmp_s0[indexin][data0inpartsort][indmantEq])/1.e3,cmap='gist_heat',s=3)
    core=plt.scatter(pos_x[index][dataindexsort][indcoreEq]/R_earth,pos_y[index][dataindexsort][indcoreEq]/R_earth,c=(tmp_s[index][dataindexsort][indcoreEq]-tmp_s0[indexin][data0inpartsort][indcoreEq])/1.e3,cmap='copper',s=3)
    cmant=plt.colorbar(mantle,cmap='gist_heat')
    ccore=plt.colorbar(core,cmap='copper')
    cmant.set_label('Mantle sp. Entropy gain (kJ/kg/K), avg = %g kJ/kg/K' % (mantSavg/1.e3),size=20)
    cmant.ax.hlines(mantSavg/1.e3,np.amin([cmant.vmin,0]),cmant.vmax,color='white')
    ccore.set_label('Core sp. Entropy gain (kJ/kg/K), avg = %g kJ/kg/K' % (coreSavg/1.e3),size=20)
    ccore.ax.hlines(coreSavg/1.e3,np.amin([ccore.vmin,0]),ccore.vmax,color='white')
    plt.gca().set_xlabel(r"X $[R_\oplus]$",size=20)
    plt.gca().set_ylabel(r"Y $[R_\oplus]$",size=20)
    plt.gca().set_xlim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_ylim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_aspect('equal')
    plt.gca().set_title('Sp. Entropy Gain'+'\n'+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Bound particles: %i"%(np.shape(index)[1]),size=20)
    plt.gcf().set_size_inches(18,12)
    plt.gca().set_facecolor('0.075')
    plt.tight_layout()
    imname = basename+'S_gain.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    # plotting temperature gain
    mantle=plt.scatter(pos_x[index][dataindexsort][indmantEq]/R_earth,pos_y[index][dataindexsort][indmantEq]/R_earth,c=(tmp_T[index][dataindexsort][indmantEq]-tmp_T0[indexin][data0inpartsort][indmantEq]),cmap='inferno',s=3)
    core=plt.scatter(pos_x[index][dataindexsort][indcoreEq]/R_earth,pos_y[index][dataindexsort][indcoreEq]/R_earth,c=(tmp_T[index][dataindexsort][indcoreEq]-tmp_T0[indexin][data0inpartsort][indcoreEq]),cmap='gist_gray',s=3)
    cmant=plt.colorbar(mantle,cmap='inferno')
    ccore=plt.colorbar(core,cmap='gist_gray')
    cmant.set_label('Mantle Temperature gain (K), avg = %g K' % mantTavg,size=20)
    cmant.ax.hlines(mantTavg,np.amin([cmant.vmin,0]),cmant.vmax,color='white')
    ccore.set_label('Core Temperature gain (K), avg = %g K' % coreTavg,size=20)
    ccore.ax.hlines(coreTavg,np.amin([ccore.vmin,0]),ccore.vmax,color='white')
    plt.gca().set_xlabel(r"X $[R_\oplus]$",size=20)
    plt.gca().set_ylabel(r"Y $[R_\oplus]$",size=20)
    plt.gca().set_xlim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_ylim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_aspect('equal')
    plt.gca().set_title('Temperature Gain'+'\n'+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Bound particles: %i"%(np.shape(index)[1]),size=20)
    plt.gcf().set_size_inches(18,12)
    plt.gca().set_facecolor('0.075')
    plt.tight_layout()
    imname = basename+'T_gain.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    ## WRITING FULL PROFILES TO TEXT FILE
    Rsort = np.argsort(rrr[index][dataindexsort][indEqPlanet])
    uEq = u[index][dataindexsort][indEqPlanet]
    deltauEq = (u[index][dataindexsort][indEqPlanet]-u0[indexin][data0inpartsort][indEqPlanet])
    SEq = tmp_s[index][dataindexsort][indEqPlanet]
    deltaSEq = (tmp_s[index][dataindexsort][indEqPlanet]-tmp_s0[indexin][data0inpartsort][indEqPlanet])
    TEq = tmp_T[index][dataindexsort][indEqPlanet]
    deltaTEq = (tmp_T[index][dataindexsort][indEqPlanet]-tmp_T0[indexin][data0inpartsort][indEqPlanet])
    PEq = P[index][dataindexsort][indEqPlanet]
    deltaPEq = (P[index][dataindexsort][indEqPlanet]-P0[indexin][data0inpartsort][indEqPlanet])
    rhoEq = rho[index][dataindexsort][indEqPlanet]
    radiusEq = rrr[index][dataindexsort][indEqPlanet]
    flatradEq = rr[index][dataindexsort][indEqPlanet]
    zEq = pos_z[index][dataindexsort][indEqPlanet]
    mat_idEq = mat_id[index][dataindexsort][indEqPlanet]
    
    radiusmeanlist=np.array([])
    u50list=np.array([])
    u25list=np.array([])
    u75list=np.array([])
    umeanlist=np.array([])
    u25meanlist=np.array([])
    u75meanlist=np.array([])
    
    deltau50list=np.array([])
    deltau25list=np.array([])
    deltau75list=np.array([])
    deltaumeanlist=np.array([])
    deltau25meanlist=np.array([])
    deltau75meanlist=np.array([])
    
    S50list=np.array([])
    S25list=np.array([])
    S75list=np.array([])
    Smeanlist=np.array([])
    S25meanlist=np.array([])
    S75meanlist=np.array([])
    
    deltaS50list=np.array([])
    deltaS25list=np.array([])
    deltaS75list=np.array([])
    deltaSmeanlist=np.array([])
    deltaS25meanlist=np.array([])
    deltaS75meanlist=np.array([])
    
    T50list=np.array([])
    T25list=np.array([])
    T75list=np.array([])
    Tmeanlist=np.array([])
    T25meanlist=np.array([])
    T75meanlist=np.array([])
    
    deltaT50list=np.array([])
    deltaT25list=np.array([])
    deltaT75list=np.array([])
    deltaTmeanlist=np.array([])
    deltaT25meanlist=np.array([])
    deltaT75meanlist=np.array([])
    
    P50list=np.array([])
    P25list=np.array([])
    P75list=np.array([])
    Pmeanlist=np.array([])
    P25meanlist=np.array([])
    P75meanlist=np.array([])
    
    deltaP50list=np.array([])
    deltaP25list=np.array([])
    deltaP75list=np.array([])
    deltaPmeanlist=np.array([])
    deltaP25meanlist=np.array([])
    deltaP75meanlist=np.array([])
    
    rho50list=np.array([])
    rho25list=np.array([])
    rho75list=np.array([])
    rhomeanlist=np.array([])
    rho25meanlist=np.array([])
    rho75meanlist=np.array([])
    
    #Lzlist=np.array([])
    
    with open(basename+'profiles.txt','w') as writefile:
        writefile.write('#R_planet'+'\n') 
        writefile.write('%.9e'%(rplanet*R_earth)+'\n')
        writefile.write('#RRR|RR|U25|U50|U75|Umean|U25mean|U75mean|deltaU25|deltaU50|deltaU75|deltaUmean|deltaU25mean|deltaU75mean|S25|S50|S75|Smean|S25mean|S75mean|deltaS25|deltaS50|delatS75|deltaSmean|deltaS25mean|deltaS75mean|T25|T50|T75|Tmean|T25mean|T75mean|deltaT25|deltaT50|deltaT75|deltaTmean|deltaT25|deltaT75|P25|P50|P75|Pmean|P25mean|P75mean|deltaP25|deltaP50|deltaP75|deltaPmean|deltaP25mean|deltaP75mean|rho25|rho50|rho75|rhomean|rho25mean|rho75mean'+'\n')
        for i in range(0,np.size(Rsort),100):
            radiusmean=np.mean(radiusEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            radiusmeanlist=np.append(radiusmeanlist,radiusmean)
            radius50=np.percentile(radiusEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            
            flatradmean=np.mean(flatradEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            flatrad50=np.percentile(flatradEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            
            u25=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            u25list=np.append(u25list,u25)
            u50=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            u50list=np.append(u50list,u50)
            u75=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            u75list=np.append(u75list,u75)
            umean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            umeanlist=np.append(umeanlist,umean)
            u25mean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=u50)])
            u25meanlist=np.append(u25meanlist,u25mean)
            u75mean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>u50)])
            u75meanlist=np.append(u75meanlist,u75mean)
            
            deltau25=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            deltau25list=np.append(deltau25list,deltau25)
            deltau50=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            deltau50list=np.append(deltau50list,deltau50)
            deltau75=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            deltau75list=np.append(deltau75list,deltau75)
            deltaumean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            deltaumeanlist=np.append(deltaumeanlist,deltaumean)
            deltau25mean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltau50)])
            deltau25meanlist=np.append(deltau25meanlist,deltau25mean)
            deltau75mean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltau50)])
            deltau75meanlist=np.append(deltau75meanlist,deltau75mean)
            
            S25=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            S25list=np.append(S25list,S25)
            S50=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            S50list=np.append(S50list,S50)
            S75=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            S75list=np.append(S75list,S75)
            Smean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            Smeanlist=np.append(Smeanlist,Smean)
            S25mean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=S50)])
            S25meanlist=np.append(S25meanlist,S25mean)
            S75mean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>S50)])
            S75meanlist=np.append(S75meanlist,S75mean)
            
            deltaS25=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            deltaS25list=np.append(deltaS25list,deltaS25)
            deltaS50=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            deltaS50list=np.append(deltaS50list,deltaS50)
            deltaS75=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            deltaS75list=np.append(deltaS75list,deltaS75)
            deltaSmean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            deltaSmeanlist=np.append(deltaSmeanlist,deltaSmean)
            deltaS25mean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaS50)])
            deltaS25meanlist=np.append(deltaS25meanlist,deltaS25mean)
            deltaS75mean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaS50)])
            deltaS75meanlist=np.append(deltaS75meanlist,deltaS75mean)
            
            T25=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            T25list=np.append(T25list,T25)
            T50=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            T50list=np.append(T50list,T50)
            T75=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            T75list=np.append(T75list,T75)
            Tmean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            Tmeanlist=np.append(Tmeanlist,Tmean)
            T25mean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=T50)])
            T25meanlist=np.append(T25meanlist,T25mean)
            T75mean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>T50)])
            T75meanlist=np.append(T75meanlist,T75mean)
            
            deltaT25=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            deltaT25list=np.append(deltaT25list,deltaT25)
            deltaT50=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            deltaT50list=np.append(deltaT50list,deltaT50)
            deltaT75=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            deltaT75list=np.append(deltaT75list,deltaT75)
            deltaTmean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            deltaTmeanlist=np.append(deltaTmeanlist,deltaTmean)
            deltaT25mean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaT50)])
            deltaT25meanlist=np.append(deltaT25meanlist,deltaT25mean)
            deltaT75mean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaT50)])
            deltaT75meanlist=np.append(deltaT75meanlist,deltaT75mean)
            
            P25=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            P25list=np.append(P25list,P25)
            P50=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            P50list=np.append(P50list,P50)
            P75=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            P75list=np.append(P75list,P75)
            Pmean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            Pmeanlist=np.append(Pmeanlist,Pmean)
            P25mean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=P50)])
            P25meanlist=np.append(P25meanlist,P25mean)
            P75mean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>P50)])
            P75meanlist=np.append(P75meanlist,P75mean)
            
            deltaP25=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            deltaP25list=np.append(deltaP25list,deltaP25)
            deltaP50=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            deltaP50list=np.append(deltaP50list,deltaP50)
            deltaP75=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            deltaP75list=np.append(deltaP75list,deltaP75)
            deltaPmean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            deltaPmeanlist=np.append(deltaPmeanlist,deltaPmean)
            deltaP25mean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaP50)])
            deltaP25meanlist=np.append(deltaP25meanlist,deltaP25mean)
            deltaP75mean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaP50)])
            deltaP75meanlist=np.append(deltaP75meanlist,deltaP75mean)
            
            rho25=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
            rho25list=np.append(rho25list,rho25)
            rho50=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
            rho50list=np.append(rho50list,rho50)
            rho75=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
            rho75list=np.append(rho75list,rho75)
            rhomean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
            rhomeanlist=np.append(rhomeanlist,rhomean)
            rho25mean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=rho50)])
            rho25meanlist=np.append(rho25meanlist,rho25mean)
            rho75mean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>rho50)])
            rho75meanlist=np.append(rho75meanlist,rho75mean)
            
            writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(radiusmean,radius50,flatradmean,flatrad50,u25,u50,u75,umean,u25mean,u75mean,deltau25,deltau50,deltau75,deltaumean,deltau25mean,deltau75mean,S25,S50,S75,Smean,S25mean,S75mean,deltaS25,deltaS50,deltaS75,deltaSmean,deltaS25mean,deltaS75mean,T25,T50,T75,Tmean,T25mean,T75mean,deltaT25,deltaT50,deltaT75,deltaTmean,deltaT25mean,deltaT75mean,P25,P50,P75,Pmean,P25mean,P75mean,deltaP25,deltaP50,deltaP75,deltaPmean,deltaP25mean,deltaP75mean,rho25,rho50,rho75,rhomean,rho25mean,rho75mean)+'\n')
    print('Saved file ',basename+'profiles.txt')
    
    # WRITING CORE PROFILES TO TEXT FILE
    Rsort = np.argsort(rrr[index][dataindexsort][indcoreEqPlanet])
    uEq = u[index][dataindexsort][indcoreEqPlanet]
    deltauEq = (u[index][dataindexsort][indcoreEqPlanet]-u0[indexin][data0inpartsort][indcoreEqPlanet])
    SEq = tmp_s[index][dataindexsort][indcoreEqPlanet]
    deltaSEq = (tmp_s[index][dataindexsort][indcoreEqPlanet]-tmp_s0[indexin][data0inpartsort][indcoreEqPlanet])
    TEq = tmp_T[index][dataindexsort][indcoreEqPlanet]
    deltaTEq = (tmp_T[index][dataindexsort][indcoreEqPlanet]-tmp_T0[indexin][data0inpartsort][indcoreEqPlanet])
    PEq = P[index][dataindexsort][indcoreEqPlanet]
    deltaPEq = (P[index][dataindexsort][indcoreEqPlanet]-P0[indexin][data0inpartsort][indcoreEqPlanet])
    rhoEq = rho[index][dataindexsort][indcoreEqPlanet]
    radiusEq = rrr[index][dataindexsort][indcoreEqPlanet]
    flatradEq = rr[index][dataindexsort][indcoreEqPlanet]
    zEq = pos_z[index][dataindexsort][indcoreEqPlanet]
    mat_idEq = mat_id[index][dataindexsort][indcoreEqPlanet]
    
    radiusmeancorelist=np.array([])
    radius50corelist=np.array([])
    flatradmeancorelist=np.array([])
    flatrad50corelist=np.array([])
    u50corelist=np.array([])
    u25corelist=np.array([])
    u75corelist=np.array([])
    umeancorelist=np.array([])
    u25meancorelist=np.array([])
    u75meancorelist=np.array([])
    
    deltau50corelist=np.array([])
    deltau25corelist=np.array([])
    deltau75corelist=np.array([])
    deltaumeancorelist=np.array([])
    deltau25meancorelist=np.array([])
    deltau75meancorelist=np.array([])
    
    S50corelist=np.array([])
    S25corelist=np.array([])
    S75corelist=np.array([])
    Smeancorelist=np.array([])
    S25meancorelist=np.array([])
    S75meancorelist=np.array([])
    
    deltaS50corelist=np.array([])
    deltaS25corelist=np.array([])
    deltaS75corelist=np.array([])
    deltaSmeancorelist=np.array([])
    deltaS25meancorelist=np.array([])
    deltaS75meancorelist=np.array([])
    
    T50corelist=np.array([])
    T25corelist=np.array([])
    T75corelist=np.array([])
    Tmeancorelist=np.array([])
    T25meancorelist=np.array([])
    T75meancorelist=np.array([])
    
    deltaT50corelist=np.array([])
    deltaT25corelist=np.array([])
    deltaT75corelist=np.array([])
    deltaTmeancorelist=np.array([])
    deltaT25meancorelist=np.array([])
    deltaT75meancorelist=np.array([])
    
    P50corelist=np.array([])
    P25corelist=np.array([])
    P75corelist=np.array([])
    Pmeancorelist=np.array([])
    P25meancorelist=np.array([])
    P75meancorelist=np.array([])
    
    deltaP50corelist=np.array([])
    deltaP25corelist=np.array([])
    deltaP75corelist=np.array([])
    deltaPmeancorelist=np.array([])
    deltaP25meancorelist=np.array([])
    deltaP75meancorelist=np.array([])
    
    rho50corelist=np.array([])
    rho25corelist=np.array([])
    rho75corelist=np.array([])
    rhomeancorelist=np.array([])
    rho25meancorelist=np.array([])
    rho75meancorelist=np.array([])
    
    Nsort = 100
    for i in range(0,np.size(Rsort),Nsort):
        radiusmean=np.mean(radiusEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])])
        radiusmeancorelist=np.append(radiusmeancorelist,radiusmean)
        radius50=np.percentile(radiusEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])],50)
        radius50corelist=np.append(radius50corelist,radius50)
        
        flatradmean=np.mean(flatradEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])])
        flatradmeancorelist=np.append(flatradmeancorelist,flatradmean)
        flatrad50=np.percentile(flatradEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])],50)
        flatrad50corelist=np.append(flatrad50corelist,flatrad50)
        
        u25=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])],25)
        u25corelist=np.append(u25corelist,u25)
        u50=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])],50)
        u50corelist=np.append(u50corelist,u50)
        u75=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])],75)
        u75corelist=np.append(u75corelist,u75)
        umean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])])
        umeancorelist=np.append(umeancorelist,umean)
        u25mean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])][np.where(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=u50)])
        u25meancorelist=np.append(u25meancorelist,u25mean)
        u75mean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+Nsort])][np.where(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>u50)])
        u75meancorelist=np.append(u75meancorelist,u75mean)
        
        deltau25=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        deltau25corelist=np.append(deltau25corelist,deltau25)
        deltau50=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        deltau50corelist=np.append(deltau50corelist,deltau50)
        deltau75=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        deltau75corelist=np.append(deltau75corelist,deltau75)
        deltaumean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        deltaumeancorelist=np.append(deltaumeancorelist,deltaumean)
        deltau25mean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltau50)])
        deltau25meancorelist=np.append(deltau25meancorelist,deltau25mean)
        deltau75mean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltau50)])
        deltau75meancorelist=np.append(deltau75meancorelist,deltau75mean)
        
        S25=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        S25corelist=np.append(S25corelist,S25)
        S50=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        S50corelist=np.append(S50corelist,S50)
        S75=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        S75corelist=np.append(S75corelist,S75)
        Smean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        Smeancorelist=np.append(Smeancorelist,Smean)
        S25mean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=S50)])
        S25meancorelist=np.append(S25meancorelist,S25mean)
        S75mean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>S50)])
        S75meancorelist=np.append(S75meancorelist,S75mean)
        
        deltaS25=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        deltaS25corelist=np.append(deltaS25corelist,deltaS25)
        deltaS50=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        deltaS50corelist=np.append(deltaS50corelist,deltaS50)
        deltaS75=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        deltaS75corelist=np.append(deltaS75corelist,deltaS75)
        deltaSmean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        deltaSmeancorelist=np.append(deltaSmeancorelist,deltaSmean)
        deltaS25mean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaS50)])
        deltaS25meancorelist=np.append(deltaS25meancorelist,deltaS25mean)
        deltaS75mean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaS50)])
        deltaS75meancorelist=np.append(deltaS75meancorelist,deltaS75mean)
        
        T25=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        T25corelist=np.append(T25corelist,T25)
        T50=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        T50corelist=np.append(T50corelist,T50)
        T75=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        T75corelist=np.append(T75corelist,T75)
        Tmean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        Tmeancorelist=np.append(Tmeancorelist,Tmean)
        T25mean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=T50)])
        T25meancorelist=np.append(T25meancorelist,T25mean)
        T75mean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>T50)])
        T75meancorelist=np.append(T75meancorelist,T75mean)
        
        deltaT25=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        deltaT25corelist=np.append(deltaT25corelist,deltaT25)
        deltaT50=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        deltaT50corelist=np.append(deltaT50corelist,deltaT50)
        deltaT75=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        deltaT75corelist=np.append(deltaT75corelist,deltaT75)
        deltaTmean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        deltaTmeancorelist=np.append(deltaTmeancorelist,deltaTmean)
        deltaT25mean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaT50)])
        deltaT25meancorelist=np.append(deltaT25meancorelist,deltaT25mean)
        deltaT75mean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaT50)])
        deltaT75meancorelist=np.append(deltaT75meancorelist,deltaT75mean)
        
        P25=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        P25corelist=np.append(P25corelist,P25)
        P50=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        P50corelist=np.append(P50corelist,P50)
        P75=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        P75corelist=np.append(P75corelist,P75)
        Pmean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        Pmeancorelist=np.append(Pmeancorelist,Pmean)
        P25mean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=P50)])
        P25meancorelist=np.append(P25meancorelist,P25mean)
        P75mean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>P50)])
        P75meancorelist=np.append(P75meancorelist,P75mean)
        
        deltaP25=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        deltaP25corelist=np.append(deltaP25corelist,deltaP25)
        deltaP50=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        deltaP50corelist=np.append(deltaP50corelist,deltaP50)
        deltaP75=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        deltaP75corelist=np.append(deltaP75corelist,deltaP75)
        deltaPmean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        deltaPmeancorelist=np.append(deltaPmeancorelist,deltaPmean)
        deltaP25mean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaP50)])
        deltaP25meancorelist=np.append(deltaP25meancorelist,deltaP25mean)
        deltaP75mean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaP50)])
        deltaP75meancorelist=np.append(deltaP75meancorelist,deltaP75mean)
        
        rho25=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        rho25corelist=np.append(rho25corelist,rho25)
        rho50=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        rho50corelist=np.append(rho50corelist,rho50)
        rho75=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        rho75corelist=np.append(rho75corelist,rho75)
        rhomean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        rhomeancorelist=np.append(rhomeancorelist,rhomean)
        rho25mean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=rho50)])
        rho25meancorelist=np.append(rho25meancorelist,rho25mean)
        rho75mean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>rho50)])
        rho75meancorelist=np.append(rho75meancorelist,rho75mean)
    
    with open(basename+'core_profiles.txt','w') as writefile:
        writefile.write('#R_planet'+'\n') 
        writefile.write('%.9e'%(rplanet*R_earth)+'\n')
        writefile.write('#RRR|RRR50|RR|RR50|U25|U50|U75|Umean|U25mean|U75mean|deltaU25|deltaU50|deltaU75|deltaUmean|deltaU25mean|deltaU75mean|S25|S50|S75|Smean|S25mean|S75mean|deltaS25|deltaS50|delatS75|deltaSmean|deltaS25mean|deltaS75mean|T25|T50|T75|Tmean|T25mean|T75mean|deltaT25|deltaT50|deltaT75|deltaTmean|deltaT25|deltaT75|P25|P50|P75|Pmean|P25mean|P75mean|deltaP25|deltaP50|deltaP75|deltaPmean|deltaP25mean|deltaP75mean|rho25|rho50|rho75|rhomean|rho25mean|rho75mean'+'\n')
        for i in range(np.size(radiusmeancorelist)):
            writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(radiusmeancorelist[i],radius50corelist[i],flatradmeancorelist[i],flatrad50corelist[i],u25corelist[i],u50corelist[i],u75corelist[i],umeancorelist[i],u25meancorelist[i],u75meancorelist[i],deltau25corelist[i],deltau50corelist[i],deltau75corelist[i],deltaumeancorelist[i],deltau25meancorelist[i],deltau75meancorelist[i],S25corelist[i],S50corelist[i],S75corelist[i],Smeancorelist[i],S25meancorelist[i],S75meancorelist[i],deltaS25corelist[i],deltaS50corelist[i],deltaS75corelist[i],deltaSmeancorelist[i],deltaS25meancorelist[i],deltaS75meancorelist[i],T25corelist[i],T50corelist[i],T75corelist[i],Tmeancorelist[i],T25meancorelist[i],T75meancorelist[i],deltaT25corelist[i],deltaT50corelist[i],deltaT75corelist[i],deltaTmeancorelist[i],deltaT25meancorelist[i],deltaT75meancorelist[i],P25corelist[i],P50corelist[i],P75corelist[i],Pmeancorelist[i],P25meancorelist[i],P75meancorelist[i],deltaP25corelist[i],deltaP50corelist[i],deltaP75corelist[i],deltaPmeancorelist[i],deltaP25meancorelist[i],deltaP75meancorelist[i],rho25corelist[i],rho50corelist[i],rho75corelist[i],rhomeancorelist[i],rho25meancorelist[i],rho75meancorelist[i])+'\n')
    print('Saved file ',basename+'core_profiles.txt')
    
    ## WRITING mantle PROFILES TO TEXT FILE
    Rsort = np.argsort(rrr[index][dataindexsort][indmantEqPlanet])
    uEq = u[index][dataindexsort][indmantEqPlanet]
    deltauEq = (u[index][dataindexsort][indmantEqPlanet]-u0[indexin][data0inpartsort][indmantEqPlanet])
    SEq = tmp_s[index][dataindexsort][indmantEqPlanet]
    deltaSEq = (tmp_s[index][dataindexsort][indmantEqPlanet]-tmp_s0[indexin][data0inpartsort][indmantEqPlanet])
    TEq = tmp_T[index][dataindexsort][indmantEqPlanet]
    deltaTEq = (tmp_T[index][dataindexsort][indmantEqPlanet]-tmp_T0[indexin][data0inpartsort][indmantEqPlanet])
    PEq = P[index][dataindexsort][indmantEqPlanet]
    deltaPEq = (P[index][dataindexsort][indmantEqPlanet]-P0[indexin][data0inpartsort][indmantEqPlanet])
    rhoEq = rho[index][dataindexsort][indmantEqPlanet]
    radiusEq = rrr[index][dataindexsort][indmantEqPlanet]
    flatradEq = rr[index][dataindexsort][indmantEqPlanet]
    zEq = pos_z[index][dataindexsort][indmantEqPlanet]
    mat_idEq = mat_id[index][dataindexsort][indmantEqPlanet]
    
    radiusmeanmantlist=np.array([])
    radius50mantlist=np.array([])
    flatradmeanmantlist=np.array([])
    flatrad50mantlist=np.array([])
    u50mantlist=np.array([])
    u25mantlist=np.array([])
    u75mantlist=np.array([])
    umeanmantlist=np.array([])
    u25meanmantlist=np.array([])
    u75meanmantlist=np.array([])
    
    deltau50mantlist=np.array([])
    deltau25mantlist=np.array([])
    deltau75mantlist=np.array([])
    deltaumeanmantlist=np.array([])
    deltau25meanmantlist=np.array([])
    deltau75meanmantlist=np.array([])
    
    S50mantlist=np.array([])
    S25mantlist=np.array([])
    S75mantlist=np.array([])
    Smeanmantlist=np.array([])
    S25meanmantlist=np.array([])
    S75meanmantlist=np.array([])
    
    deltaS50mantlist=np.array([])
    deltaS25mantlist=np.array([])
    deltaS75mantlist=np.array([])
    deltaSmeanmantlist=np.array([])
    deltaS25meanmantlist=np.array([])
    deltaS75meanmantlist=np.array([])
    
    T50mantlist=np.array([])
    T25mantlist=np.array([])
    T75mantlist=np.array([])
    Tmeanmantlist=np.array([])
    T25meanmantlist=np.array([])
    T75meanmantlist=np.array([])
    
    deltaT50mantlist=np.array([])
    deltaT25mantlist=np.array([])
    deltaT75mantlist=np.array([])
    deltaTmeanmantlist=np.array([])
    deltaT25meanmantlist=np.array([])
    deltaT75meanmantlist=np.array([])
    
    P50mantlist=np.array([])
    P25mantlist=np.array([])
    P75mantlist=np.array([])
    Pmeanmantlist=np.array([])
    P25meanmantlist=np.array([])
    P75meanmantlist=np.array([])
    
    P50fitmantlist=np.array([])
    P25fitmantlist=np.array([])
    P75fitmantlist=np.array([])
    Pmeanfitmantlist=np.array([])
    P25meanfitmantlist=np.array([])
    P75meanfitmantlist=np.array([])
    
    deltaP50mantlist=np.array([])
    deltaP25mantlist=np.array([])
    deltaP75mantlist=np.array([])
    deltaPmeanmantlist=np.array([])
    deltaP25meanmantlist=np.array([])
    deltaP75meanmantlist=np.array([])
    
    rho50mantlist=np.array([])
    rho25mantlist=np.array([])
    rho75mantlist=np.array([])
    rhomeanmantlist=np.array([])
    rho25meanmantlist=np.array([])
    rho75meanmantlist=np.array([])
    
    for i in range(0,np.size(Rsort),100):
        radiusmean=np.mean(radiusEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        radiusmeanmantlist=np.append(radiusmeanmantlist,radiusmean)
        radius50=np.percentile(radiusEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        radius50mantlist=np.append(radius50mantlist,radius50)
        
        flatradmean=np.mean(flatradEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        flatradmeanmantlist=np.append(flatradmeanmantlist,flatradmean)
        flatrad50=np.percentile(flatradEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        flatrad50mantlist=np.append(flatrad50mantlist,flatrad50)
        
        u25=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        u25mantlist=np.append(u25mantlist,u25)
        u50=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        u50mantlist=np.append(u50mantlist,u50)
        u75=np.percentile(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        u75mantlist=np.append(u75mantlist,u75)
        umean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        umeanmantlist=np.append(umeanmantlist,umean)
        u25mean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=u50)])
        u25meanmantlist=np.append(u25meanmantlist,u25mean)
        u75mean=np.mean(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(uEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>u50)])
        u75meanmantlist=np.append(u75meanmantlist,u75mean)
        
        deltau25=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        deltau25mantlist=np.append(deltau25mantlist,deltau25)
        deltau50=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        deltau50mantlist=np.append(deltau50mantlist,deltau50)
        deltau75=np.percentile(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        deltau75mantlist=np.append(deltau75mantlist,deltau75)
        deltaumean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        deltaumeanmantlist=np.append(deltaumeanmantlist,deltaumean)
        deltau25mean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltau50)])
        deltau25meanmantlist=np.append(deltau25meanmantlist,deltau25mean)
        deltau75mean=np.mean(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltauEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltau50)])
        deltau75meanmantlist=np.append(deltau75meanmantlist,deltau75mean)
        
        S25=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        S25mantlist=np.append(S25mantlist,S25)
        S50=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        S50mantlist=np.append(S50mantlist,S50)
        S75=np.percentile(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        S75mantlist=np.append(S75mantlist,S75)
        Smean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        Smeanmantlist=np.append(Smeanmantlist,Smean)
        S25mean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=S50)])
        S25meanmantlist=np.append(S25meanmantlist,S25mean)
        S75mean=np.mean(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(SEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>S50)])
        S75meanmantlist=np.append(S75meanmantlist,S75mean)
        
        deltaS25=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        deltaS25mantlist=np.append(deltaS25mantlist,deltaS25)
        deltaS50=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        deltaS50mantlist=np.append(deltaS50mantlist,deltaS50)
        deltaS75=np.percentile(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        deltaS75mantlist=np.append(deltaS75mantlist,deltaS75)
        deltaSmean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        deltaSmeanmantlist=np.append(deltaSmeanmantlist,deltaSmean)
        deltaS25mean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaS50)])
        deltaS25meanmantlist=np.append(deltaS25meanmantlist,deltaS25mean)
        deltaS75mean=np.mean(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaSEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaS50)])
        deltaS75meanmantlist=np.append(deltaS75meanmantlist,deltaS75mean)
        
        T25=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        T25mantlist=np.append(T25mantlist,T25)
        T50=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        T50mantlist=np.append(T50mantlist,T50)
        T75=np.percentile(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        T75mantlist=np.append(T75mantlist,T75)
        Tmean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        Tmeanmantlist=np.append(Tmeanmantlist,Tmean)
        T25mean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=T50)])
        T25meanmantlist=np.append(T25meanmantlist,T25mean)
        T75mean=np.mean(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(TEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>T50)])
        T75meanmantlist=np.append(T75meanmantlist,T75mean)
        
        deltaT25=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        deltaT25mantlist=np.append(deltaT25mantlist,deltaT25)
        deltaT50=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        deltaT50mantlist=np.append(deltaT50mantlist,deltaT50)
        deltaT75=np.percentile(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        deltaT75mantlist=np.append(deltaT75mantlist,deltaT75)
        deltaTmean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        deltaTmeanmantlist=np.append(deltaTmeanmantlist,deltaTmean)
        deltaT25mean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaT50)])
        deltaT25meanmantlist=np.append(deltaT25meanmantlist,deltaT25mean)
        deltaT75mean=np.mean(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaTEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaT50)])
        deltaT75meanmantlist=np.append(deltaT75meanmantlist,deltaT75mean)
        
        P25=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        P25mantlist=np.append(P25mantlist,P25)
        P50=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        P50mantlist=np.append(P50mantlist,P50)
        P75=np.percentile(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        P75mantlist=np.append(P75mantlist,P75)
        Pmean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        Pmeanmantlist=np.append(Pmeanmantlist,Pmean)
        P25mean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=P50)])
        P25meanmantlist=np.append(P25meanmantlist,P25mean)
        P75mean=np.mean(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(PEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>P50)])
        P75meanmantlist=np.append(P75meanmantlist,P75mean)
        
        deltaP25=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        deltaP25mantlist=np.append(deltaP25mantlist,deltaP25)
        deltaP50=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        deltaP50mantlist=np.append(deltaP50mantlist,deltaP50)
        deltaP75=np.percentile(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        deltaP75mantlist=np.append(deltaP75mantlist,deltaP75)
        deltaPmean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        deltaPmeanmantlist=np.append(deltaPmeanmantlist,deltaPmean)
        deltaP25mean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=deltaP50)])
        deltaP25meanmantlist=np.append(deltaP25meanmantlist,deltaP25mean)
        deltaP75mean=np.mean(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(deltaPEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>deltaP50)])
        deltaP75meanmantlist=np.append(deltaP75meanmantlist,deltaP75mean)
        
        rho25=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],25)
        rho25mantlist=np.append(rho25mantlist,rho25)
        rho50=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],50)
        rho50mantlist=np.append(rho50mantlist,rho50)
        rho75=np.percentile(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])],75)
        rho75mantlist=np.append(rho75mantlist,rho75)
        rhomean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])])
        rhomeanmantlist=np.append(rhomeanmantlist,rhomean)
        rho25mean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])]<=rho50)])
        rho25meanmantlist=np.append(rho25meanmantlist,rho25mean)
        rho75mean=np.mean(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])][np.where(rhoEq[Rsort][i:np.amin([np.size(Rsort),i+100])]>rho50)])
        rho75meanmantlist=np.append(rho75meanmantlist,rho75mean)
    
    with open(basename+'mantle_profiles.txt','w') as writefile:
        writefile.write('#R_planet'+'\n') 
        writefile.write('%.9e'%(rplanet*R_earth)+'\n')
        writefile.write('#RRR|RRR50|RR|RR50|U25|U50|U75|Umean|U25mean|U75mean|deltaU25|deltaU50|deltaU75|deltaUmean|deltaU25mean|deltaU75mean|S25|S50|S75|Smean|S25mean|S75mean|deltaS25|deltaS50|delatS75|deltaSmean|deltaS25mean|deltaS75mean|T25|T50|T75|Tmean|T25mean|T75mean|deltaT25|deltaT50|deltaT75|deltaTmean|deltaT25|deltaT75|P25|P50|P75|Pmean|P25mean|P75mean|deltaP25|deltaP50|deltaP75|deltaPmean|deltaP25mean|deltaP75mean|rho25|rho50|rho75|rhomean|rho25mean|rho75mean'+'\n')
        for i in range(np.size(radiusmeanmantlist)):
            writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(radiusmeanmantlist[i],radius50mantlist[i],flatradmeanmantlist[i],flatrad50mantlist[i],u25mantlist[i],u50mantlist[i],u75mantlist[i],umeanmantlist[i],u25meanmantlist[i],u75meanmantlist[i],deltau25mantlist[i],deltau50mantlist[i],deltau75mantlist[i],deltaumeanmantlist[i],deltau25meanmantlist[i],deltau75meanmantlist[i],S25mantlist[i],S50mantlist[i],S75mantlist[i],Smeanmantlist[i],S25meanmantlist[i],S75meanmantlist[i],deltaS25mantlist[i],deltaS50mantlist[i],deltaS75mantlist[i],deltaSmeanmantlist[i],deltaS25meanmantlist[i],deltaS75meanmantlist[i],T25mantlist[i],T50mantlist[i],T75mantlist[i],Tmeanmantlist[i],T25meanmantlist[i],T75meanmantlist[i],deltaT25mantlist[i],deltaT50mantlist[i],deltaT75mantlist[i],deltaTmeanmantlist[i],deltaT25meanmantlist[i],deltaT75meanmantlist[i],P25mantlist[i],P50mantlist[i],P75mantlist[i],Pmeanmantlist[i],P25meanmantlist[i],P75meanmantlist[i],deltaP25mantlist[i],deltaP50mantlist[i],deltaP75mantlist[i],deltaPmeanmantlist[i],deltaP25meanmantlist[i],deltaP75meanmantlist[i],rho25mantlist[i],rho50mantlist[i],rho75mantlist[i],rhomeanmantlist[i],rho25meanmantlist[i],rho75meanmantlist[i])+'\n')
    print('Saved file ',basename+'mantle_profiles.txt')
    
    rCMB=np.amin(radiusmeanmantlist)/R_earth
    vP_rubie=np.vectorize(P_rubie,otypes=['float64'])
    vT_rubie=np.vectorize(T_rubie,otypes=['float64'])
    cmb_rubie = CMB_rubie(mtot/M_earth,mcore,mmant)
    
    # PRESSURE PROFILE FITS FOR 25TH PERCENTILE
    argmantfitmaxP25 = np.amax(np.where(P25mantlist>=1e9))
    mantfitrangeP25 = radiusmeanmantlist[argmantfitmaxP25]-radiusmeanmantlist[0]
    mantfitP25 = np.where((radiusmeanmantlist-radiusmeanmantlist[0])/mantfitrangeP25>=0.15)
    P25mantpolyfit = np.polyfit(radiusmeanmantlist[mantfitP25],P25mantlist[mantfitP25],5)
    P25mantfitfn = np.poly1d(P25mantpolyfit)
    P25extendfit = np.polyfit(radiusmeanmantlist[mantfitP25][:2],P25mantfitfn(radiusmeanmantlist[mantfitP25][:2]),1)
    P25extendfn = np.poly1d(P25extendfit)
    
    corefitP25 =np.where((radiusmeancorelist)/(radiusmeanmantlist[0])<=0.85)
    P25corepolyfit = np.polyfit(radiusmeancorelist[corefitP25],P25corelist[corefitP25],4)
    P25corefitfn = np.poly1d(P25corepolyfit)
    P25extendcorefit = np.polyfit(radiusmeancorelist[corefitP25][-3:],P25corefitfn(radiusmeancorelist[corefitP25][-3:]),1)
    P25extendcorefn = np.poly1d(P25extendcorefit)
    P25extendfit = np.where((radiusmeanlist<=radiusmeanmantlist[mantfitP25][10])&(radiusmeanlist>radiusmeancorelist[corefitP25][-1]))
    
    ## establishing ranges for various pressure profile functions
    ### intersection of the two extended linear fits
    P25linFail=False
    P25fitFail=False
    P25corefitFail=False
    P25fitlist = np.zeros(np.size(radiusmeanlist))
    P25modellist = np.zeros(np.size(radiusmeanlist))
    idxlinP25 = np.argwhere(np.diff(np.sign(P25extendcorefn(radiusmeanlist)-P25extendfn(radiusmeanlist)))).flatten()
    if np.size(idxlinP25)==0:
        idxlinP25=np.array([np.size(radiusmeanlist)-1])
        P25linFail=True
        print('P25linFAIL')
    idxfitP25 = np.argwhere(np.diff(np.sign(P25corefitfn(radiusmeanlist)-P25extendfn(radiusmeanlist)))).flatten()
    if np.size(idxfitP25)==0:
        idxfitP25=corefitP25[-1]
        P25fitFail=True
        print('P25fitFAIL')
    idxcorefitP25 = np.argwhere(np.diff(np.sign(P25extendcorefn(radiusmeanlist)-P25mantfitfn(radiusmeanlist)))).flatten()
    if np.size(idxcorefitP25)==0:
        idxcorefitP25=mantfitP25[0]
        P25corefitFail=True
        print('P25corefitFAIL')
    ### place in radius function where we fit core fit function
    #### if the two linear extensions don't intersect between the fit regions, use the mantle extension-core fit intersect
    if (radiusmeanlist[idxlinP25]<radiusmeancorelist[corefitP25][-3])|(radiusmeanlist[idxlinP25]>radiusmeanmantlist[mantfitP25][2]):
        if P25linFail:    
            wherecorefit = np.where(radiusmeanlist<=radiusmeanlist[idxfitP25][0])
            wherecoreextfit = np.where(radiusmeanlist<0)
            wheremantextfit = np.where((radiusmeanlist>radiusmeanlist[idxfitP25][0]) & (radiusmeanlist<radiusmeanmantlist[mantfitP25][2]))
            wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP25][2])
        elif P25fitFail:
            wherecorefit = np.where(radiusmeanlist<radiusmeancorelist[corefitP25][-3])
            wherecoreextfit = np.where((radiusmeanlist>=radiusmeancorelist[corefitP25][-3]) & (radiusmeanlist<radiusmeanlist[idxcorefitP25][-1]))
            wheremantextfit = np.where(radiusmeanlist<0)
            wheremantfit = np.where(radiusmeanlist>=radiusmeanlist[idxcorefitP25][-1])
        else:
            wherecorefit = np.where(radiusmeanlist<=radiusmeanlist[idxfitP25][0])
            wherecoreextfit = np.where(radiusmeanlist<0)
            wheremantextfit = np.where((radiusmeanlist>radiusmeanlist[idxfitP25][0]) & (radiusmeanlist<radiusmeanmantlist[mantfitP25][2]))
            wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP25][2])
    else:
        wherecorefit = np.where(radiusmeanlist<radiusmeancorelist[corefitP25][-3])
        wherecoreextfit = np.where((radiusmeanlist>=radiusmeancorelist[corefitP25][-3]) & (radiusmeanlist<radiusmeanlist[idxlinP25]))
        wheremantextfit = np.where((radiusmeanlist>=radiusmeanlist[idxlinP25]) & (radiusmeanlist<radiusmeanmantlist[mantfitP25][2]))
        wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP25][2])
    ## adding in fit values
    P25fitlist[wherecorefit]=P25corefitfn(radiusmeanlist[wherecorefit])
    P25modellist[wherecorefit]=0
    P25fitlist[wherecoreextfit]=P25extendcorefn(radiusmeanlist[wherecoreextfit])
    P25modellist[wherecoreextfit]=1
    P25fitlist[wheremantextfit]=P25extendfn(radiusmeanlist[wheremantextfit])
    P25modellist[wheremantextfit]=2
    P25fitlist[wheremantfit]=P25mantfitfn(radiusmeanlist[wheremantfit])
    P25modellist[wheremantfit]=3
    
    # PRESSURE PROFILE FITS FOR 50TH PERCENTILE
    argmantfitmaxP50 = np.amax(np.where(P50mantlist>=1e9))
    mantfitrangeP50 = radiusmeanmantlist[argmantfitmaxP50]-radiusmeanmantlist[0]
    mantfitP50 = np.where((radiusmeanmantlist-radiusmeanmantlist[0])/mantfitrangeP50>=0.15)
    P50mantpolyfit = np.polyfit(radiusmeanmantlist[mantfitP50],P50mantlist[mantfitP50],5)
    P50mantfitfn = np.poly1d(P50mantpolyfit)
    P50extendfit = np.polyfit(radiusmeanmantlist[mantfitP50][:2],P50mantfitfn(radiusmeanmantlist[mantfitP50][:2]),1)
    P50extendfn = np.poly1d(P50extendfit)
    
    corefitP50 = np.where((radiusmeancorelist)/(radiusmeanmantlist[0])<=0.85)
    P50corepolyfit = np.polyfit(radiusmeancorelist[corefitP50],P50corelist[corefitP50],4)
    P50corefitfn = np.poly1d(P50corepolyfit)
    P50extendcorefit = np.polyfit(radiusmeancorelist[corefitP50][-3:],P50corefitfn(radiusmeancorelist[corefitP50][-3:]),1)
    P50extendcorefn = np.poly1d(P50extendcorefit)
    P50extendfit = np.where((radiusmeanlist<=radiusmeanmantlist[mantfitP50][10])&(radiusmeanlist>radiusmeancorelist[corefitP50][-1]))
    
    ## establishing ranges for various pressure profile functions
    ### intersection of the two extended linear fits
    P50linFail=False
    P50fitFail=False
    P50corefitFail=False
    P50fitlist = np.zeros(np.size(radiusmeanlist))
    P50modellist = np.zeros(np.size(radiusmeanlist))
    idxlinP50 = np.argwhere(np.diff(np.sign(P50extendcorefn(radiusmeanlist)-P50extendfn(radiusmeanlist)))).flatten()
    if np.size(idxlinP50)==0:
        idxlinP50=np.array([np.size(radiusmeanlist)-1])
        P50linFail=True
        print('P50linFAIL')
    idxfitP50 = np.argwhere(np.diff(np.sign(P50corefitfn(radiusmeanlist)-P50extendfn(radiusmeanlist)))).flatten()
    if np.size(idxfitP50)==0:
        idxfitP50=corefitP50[-1]
        P50fitFail=True
        print('P50fitFAIL')
    idxcorefitP50 = np.argwhere(np.diff(np.sign(P50extendcorefn(radiusmeanlist)-P50mantfitfn(radiusmeanlist)))).flatten()
    if np.size(idxcorefitP50)==0:
        idxcorefitP50=mantfitP50[0]
        P50corefitFail=True
        print('P50corefitFAIL')
    ### place in radius function where we fit core fit function
    #### if the two linear extensions don't intersect between the fit regions, use the mantle extension-core fit intersect
    if (radiusmeanlist[idxlinP50]<radiusmeancorelist[corefitP50][-3])|(radiusmeanlist[idxlinP50]>radiusmeanmantlist[mantfitP50][2]):
        if P50linFail:    
            wherecorefit = np.where(radiusmeanlist<=radiusmeanlist[idxfitP50][0])
            wherecoreextfit = np.where(radiusmeanlist<0)
            wheremantextfit = np.where((radiusmeanlist>radiusmeanlist[idxfitP50][0]) & (radiusmeanlist<radiusmeanmantlist[mantfitP50][2]))
            wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP50][2])
        elif P50fitFail:
            wherecorefit = np.where(radiusmeanlist<radiusmeancorelist[corefitP50][-3])
            wherecoreextfit = np.where((radiusmeanlist>=radiusmeancorelist[corefitP50][-3]) & (radiusmeanlist<radiusmeanlist[idxcorefitP50][-1]))
            wheremantextfit = np.where(radiusmeanlist<0)
            wheremantfit = np.where(radiusmeanlist>=radiusmeanlist[idxcorefitP50][-1])
        else:
            wherecorefit = np.where(radiusmeanlist<=radiusmeanlist[idxfitP50][0])
            wherecoreextfit = np.where(radiusmeanlist<0)
            wheremantextfit = np.where((radiusmeanlist>radiusmeanlist[idxfitP50][0]) & (radiusmeanlist<radiusmeanmantlist[mantfitP50][2]))
            wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP50][2])
    else:
        wherecorefit = np.where(radiusmeanlist<radiusmeancorelist[corefitP50][-3])
        wherecoreextfit = np.where((radiusmeanlist>=radiusmeancorelist[corefitP50][-3]) & (radiusmeanlist<radiusmeanlist[idxlinP50]))
        wheremantextfit = np.where((radiusmeanlist>=radiusmeanlist[idxlinP50]) & (radiusmeanlist<radiusmeanmantlist[mantfitP50][2]))
        wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP50][2])
    ## adding in fit values
    P50fitlist[wherecorefit]=P50corefitfn(radiusmeanlist[wherecorefit])
    P50modellist[wherecorefit]=0
    P50fitlist[wherecoreextfit]=P50extendcorefn(radiusmeanlist[wherecoreextfit])
    P50modellist[wherecoreextfit]=1
    P50fitlist[wheremantextfit]=P50extendfn(radiusmeanlist[wheremantextfit])
    P50modellist[wheremantextfit]=2
    P50fitlist[wheremantfit]=P50mantfitfn(radiusmeanlist[wheremantfit])
    P50modellist[wheremantfit]=3
    
    # PRESSURE PROFILE FITS FOR 75TH PERCENTILE
    argmantfitmaxP75 = np.amax(np.where(P75mantlist>=1e9))
    mantfitrangeP75 = radiusmeanmantlist[argmantfitmaxP75]-radiusmeanmantlist[0]
    mantfitP75 = np.where((radiusmeanmantlist-radiusmeanmantlist[0])/mantfitrangeP75>=0.15)
    P75mantpolyfit = np.polyfit(radiusmeanmantlist[mantfitP75],P75mantlist[mantfitP75],5)
    P75mantfitfn = np.poly1d(P75mantpolyfit)
    P75extendfit = np.polyfit(radiusmeanmantlist[mantfitP75][:2],P75mantfitfn(radiusmeanmantlist[mantfitP75][:2]),1)
    P75extendfn = np.poly1d(P75extendfit)
    
    corefitP75 = np.where((radiusmeancorelist)/(radiusmeanmantlist[0])<=0.85)
    P75corepolyfit = np.polyfit(radiusmeancorelist[corefitP75],P75corelist[corefitP75],4)
    P75corefitfn = np.poly1d(P75corepolyfit)
    P75extendcorefit = np.polyfit(radiusmeancorelist[corefitP75][-3:],P75corefitfn(radiusmeancorelist[corefitP75][-3:]),1)
    P75extendcorefn = np.poly1d(P75extendcorefit)
    P75extendfit = np.where((radiusmeanlist<=radiusmeanmantlist[mantfitP75][0])&(radiusmeanlist>radiusmeancorelist[corefitP75][-1]))
    
    ## establishing ranges for various pressure profile functions
    ### intersection of the two extended linear fits
    P75linFail=False
    P75fitFail=False
    P75corefitFail=False
    P75fitlist = np.zeros(np.size(radiusmeanlist))
    P75modellist = np.zeros(np.size(radiusmeanlist))
    idxlinP75 = np.argwhere(np.diff(np.sign(P75extendcorefn(radiusmeanlist)-P75extendfn(radiusmeanlist)))).flatten()
    if np.size(idxlinP75)==0:
        idxlinP75=np.array([np.size(radiusmeanlist)-1])
        P75linFail=True
        print('P75linFAIL')
    idxfitP75 = np.argwhere(np.diff(np.sign(P75corefitfn(radiusmeanlist)-P75extendfn(radiusmeanlist)))).flatten()
    if np.size(idxfitP75)==0:
        idxfitP75=corefitP75[-1]
        P75fitFail=True
        print('P75fitFAIL')
    idxcorefitP75 = np.argwhere(np.diff(np.sign(P75extendcorefn(radiusmeanlist)-P75mantfitfn(radiusmeanlist)))).flatten()
    if np.size(idxcorefitP75)==0:
        idxcorefitP75=mantfitP75[0]
        P75corefitFail=True
        print('P75corefitFAIL')
    if (P75linFail & P75fitFail & P75corefitFail):
        P75patchfit = np.polyfit([radiusmeancorelist[corefitP75][-1],radiusmeanmantlist[mantfitP75][0]],[P75corefitfn(radiusmeancorelist[corefitP75][-1]),P75mantfitfn(radiusmeanmantlist[mantfitP75][0])],1)
        P75patchfn = np.poly1d(P75patchfit)
        wherecorefit = np.where(radiusmeanlist<radiusmeancorelist[corefitP75][-3])
        wherepatchfit = np.where((radiusmeanlist>=radiusmeancorelist[corefitP75][-3])&(radiusmeanlist<radiusmeanmantlist[mantfitP75][2]))
        wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP75][2])
    ### place in radius function where we fit core fit function
    #### if the two linear extensions don't intersect between the fit regions, use the mantle extension-core fit intersect
    elif (radiusmeanlist[idxlinP75]<radiusmeancorelist[corefitP75][-3])|(radiusmeanlist[idxlinP75]>radiusmeanmantlist[mantfitP75][2]):
        if P75linFail:    
            wherecorefit = np.where(radiusmeanlist<=radiusmeanlist[idxfitP75][0])
            wherecoreextfit = np.where(radiusmeanlist<0)
            wheremantextfit = np.where((radiusmeanlist>radiusmeanlist[idxfitP75][0]) & (radiusmeanlist<radiusmeanmantlist[mantfitP75][2]))
            wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP75][2])
        elif P75fitFail:
            wherecorefit = np.where(radiusmeanlist<radiusmeancorelist[corefitP75][-3])
            wherecoreextfit = np.where((radiusmeanlist>=radiusmeancorelist[corefitP75][-3]) & (radiusmeanlist<radiusmeanlist[idxcorefitP75][-1]))
            wheremantextfit = np.where(radiusmeanlist<0)
            wheremantfit = np.where(radiusmeanlist>=radiusmeanlist[idxcorefitP75][-1])
        else:
            wherecorefit = np.where(radiusmeanlist<=radiusmeanlist[idxfitP75][0])
            wherecoreextfit = np.where(radiusmeanlist<0)
            wheremantextfit = np.where((radiusmeanlist>radiusmeanlist[idxfitP75][0]) & (radiusmeanlist<radiusmeanmantlist[mantfitP75][2]))
            wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP75][2])
    else:
        wherecorefit = np.where(radiusmeanlist<radiusmeancorelist[corefitP75][-3])
        wherecoreextfit = np.where((radiusmeanlist>=radiusmeancorelist[corefitP75][-3]) & (radiusmeanlist<radiusmeanlist[idxlinP75]))
        wheremantextfit = np.where((radiusmeanlist>=radiusmeanlist[idxlinP75]) & (radiusmeanlist<radiusmeanmantlist[mantfitP75][2]))
        wheremantfit = np.where(radiusmeanlist>=radiusmeanmantlist[mantfitP75][2])
    ## adding in fit values
    if (P75linFail & P75fitFail & P75corefitFail):
        P75fitlist[wherecorefit]=P75corefitfn(radiusmeanlist[wherecorefit])
        P75modellist[wherecorefit]=0
        P75fitlist[wheremantfit]=P75mantfitfn(radiusmeanlist[wheremantfit])
        P75modellist[wheremantfit]=3
        P75fitlist[wherepatchfit]=P75patchfn(radiusmeanlist[wherepatchfit])
        P75modellist[wherepatchfit]=5
    else:
        P75fitlist[wherecorefit]=P75corefitfn(radiusmeanlist[wherecorefit])
        P75modellist[wherecorefit]=0
        P75fitlist[wherecoreextfit]=P75extendcorefn(radiusmeanlist[wherecoreextfit])
        P75modellist[wherecoreextfit]=1
        P75fitlist[wheremantextfit]=P75extendfn(radiusmeanlist[wheremantextfit])
        P75modellist[wheremantextfit]=2
        P75fitlist[wheremantfit]=P75mantfitfn(radiusmeanlist[wheremantfit])
        P75modellist[wheremantfit]=3
    
    with open(basename+'pressure_profiles.txt','w') as writefile:
        #writefile.write('#R_planet'+'\n') 
        #writefile.write('%.9e'%(rplanet*R_earth)+'\n')
        writefile.write('#RRR|P25fit|P50fit|P75fit|P25model|P50model|P75model'+'\n')
        for i in range(np.size(radiusmeanlist)):
            writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:1f}|{:1f}|{:1f}'.format(radiusmeanlist[i],P25fitlist[i],P50fitlist[i],P75fitlist[i],P25modellist[i],P50modellist[i],P75modellist[i])+'\n')
    print('Saved file ',basename+'pressure_profiles.txt')
    
    if os.path.isfile(basename+'final'):
        HERCparams=HERCULES_parameters()
        HERCp=HERCULES_planet()
        with open(basename+'final', "rb") as file:
            HERCparams.read_binary(file)
            HERCp.read_binary(file)
        HERCp.calc_pCMB()
        HERC_PCMB = HERCp.pCMB
        HERC_Lz = np.abs(HERCp.Lout[-1]/LEM)
        HERC_r=np.array([])
        HERC_P=np.array([])
        for i in range(HERCp.Nlayer):
            HERC_r = np.append(HERC_r,HERCp.layers[-1-i].a)
            HERC_P = np.append(HERC_P,HERCp.press[-1-i])
            
        core_layer=HERCp.Nmaterial-1
        temp=np.where(HERCp.flag_material==core_layer)[0]
        ind=temp[0]
        HERCc_layer=HERCp.layers[ind]

    solvusfn = interpolate.interp1d([0,50.e9,100.e9,400.e9,1000.e9],[4080,6094,6752,9337,14507.0])    

    fig, ax = plt.subplots(3, 3, figsize=(18,18))
    corecolor='dimgray'
    mantlecolor='indianred'
    ax[0,0].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,rho[index][dataindexsort][indcoreEqPlanet],s=2,color=corecolor)
    ax[0,0].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,rho[index][dataindexsort][indmantEqPlanet],s=2,color=mantlecolor)
    ax[0,0].plot(radiusmeanlist/R_earth,rho25list)
    ax[0,0].plot(radiusmeanlist/R_earth,rho50list)
    ax[0,0].plot(radiusmeanlist/R_earth,rho75list)
    ax[0,0].plot(radiusmeanlist/R_earth,rho25meanlist)
    ax[0,0].plot(radiusmeanlist/R_earth,rhomeanlist)
    ax[0,0].plot(radiusmeanlist/R_earth,rho75meanlist)
    ax[0,0].set_xlabel(r"R $[R_\oplus]$")
    ax[0,0].set_ylabel(r'Density $rho [g/cm^3]$')
    ax[0,0].tick_params(axis='both', which='both', labelsize=18)
    ax[0,0].set_yscale('log')
    
    ax[0,1].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,P[index][dataindexsort][indcoreEqPlanet]/1.e9,s=2,color=corecolor)
    ax[0,1].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,P[index][dataindexsort][indmantEqPlanet]/1.e9,s=2,color=mantlecolor)
    ax[0,1].plot(radiusmeanlist/R_earth,P25list/1.e9)
    ax[0,1].plot(radiusmeanlist/R_earth,P50list/1.e9)
    ax[0,1].plot(radiusmeanlist/R_earth,P75list/1.e9)
    ax[0,1].plot(radiusmeanlist/R_earth,P25meanlist/1.e9)
    ax[0,1].plot(radiusmeanlist/R_earth,Pmeanlist/1.e9)
    ax[0,1].plot(radiusmeanlist/R_earth,P75meanlist/1.e9)
    #ax[0,1].plot(radiusmeanlist/R_earth,vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth)/1.e9,color='black')
    ax[0,1].set_xlabel(r"R $[R_\oplus]$")
    ax[0,1].set_ylabel('Pressure P [GPa]')
    ax[0,1].tick_params(axis='both', which='both', labelsize=18)
    #ax[0,1].set_yscale('log')
    
    ax[0,2].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,(P[index][dataindexsort][indcoreEqPlanet]-P0[indexin][data0inpartsort][indcoreEqPlanet])/1.e9,s=2,color=corecolor)
    ax[0,2].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,(P[index][dataindexsort][indmantEqPlanet]-P0[indexin][data0inpartsort][indmantEqPlanet])/1.e9,s=2,color=mantlecolor)
    ax[0,2].plot(radiusmeanlist/R_earth,deltaP25list/1.e9)
    ax[0,2].plot(radiusmeanlist/R_earth,deltaP50list/1.e9)
    ax[0,2].plot(radiusmeanlist/R_earth,deltaP75list/1.e9)
    ax[0,2].plot(radiusmeanlist/R_earth,deltaP25meanlist/1.e9)
    ax[0,2].plot(radiusmeanlist/R_earth,deltaPmeanlist/1.e9)
    ax[0,2].plot(radiusmeanlist/R_earth,deltaP75meanlist/1.e9)
    ax[0,2].set_xlabel(r"R $[R_\oplus]$")
    ax[0,2].set_ylabel(r'Pressure change $\Delta P$ [GPa]')
    ax[0,2].tick_params(axis='both', which='both', labelsize=18)
    
    ax[1,0].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,u[index][dataindexsort][indcoreEqPlanet]/1.e6,s=2,color=corecolor)
    ax[1,0].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,u[index][dataindexsort][indmantEqPlanet]/1.e6,s=2,color=mantlecolor)
    ax[1,0].plot(radiusmeancorelist[:-1]/R_earth,u25corelist[:-1]/1.e6,color='tab:cyan')
    ax[1,0].plot(radiusmeancorelist[:-1]/R_earth,u50corelist[:-1]/1.e6,color='tab:blue')
    ax[1,0].plot(radiusmeancorelist[:-1]/R_earth,u75corelist[:-1]/1.e6,color='tab:purple')
    ax[1,0].plot(radiusmeancorelist[:-1]/R_earth,u25meancorelist[:-1]/1.e6,color='c')
    ax[1,0].plot(radiusmeancorelist[:-1]/R_earth,umeancorelist[:-1]/1.e6,color='b')
    ax[1,0].plot(radiusmeancorelist[:-1]/R_earth,u75meancorelist[:-1]/1.e6,color='m')
    ax[1,0].plot(radiusmeanmantlist/R_earth,u25mantlist/1.e6,color='tab:olive')
    ax[1,0].plot(radiusmeanmantlist/R_earth,u50mantlist/1.e6,color='tab:green')
    ax[1,0].plot(radiusmeanmantlist/R_earth,u75mantlist/1.e6,color='tab:brown')
    ax[1,0].plot(radiusmeanmantlist/R_earth,u25meanmantlist/1.e6,color='y')
    ax[1,0].plot(radiusmeanmantlist/R_earth,umeanmantlist/1.e6,color='g')
    ax[1,0].plot(radiusmeanmantlist/R_earth,u75meanmantlist/1.e6,color='brown')
    ax[1,0].set_xlabel(r"R $[R_\oplus]$")
    ax[1,0].set_ylabel('Sp. energy U [MJ/kg]')
    ax[1,0].tick_params(axis='both', which='both', labelsize=18)
    
    ax[1,1].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,tmp_s[index][dataindexsort][indcoreEqPlanet]/1.e3,s=2,color=corecolor)
    ax[1,1].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,tmp_s[index][dataindexsort][indmantEqPlanet]/1.e3,s=2,color=mantlecolor)
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,S25corelist[:-1]/1.e3,color='tab:cyan')
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,S50corelist[:-1]/1.e3,color='tab:blue')
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,S75corelist[:-1]/1.e3,color='tab:purple')
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,S25meancorelist[:-1]/1.e3,color='c')
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,Smeancorelist[:-1]/1.e3,color='b')
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,S75meancorelist[:-1]/1.e3,color='m')
    ax[1,1].plot(radiusmeanmantlist/R_earth,S25mantlist/1.e3,color='tab:olive')
    ax[1,1].plot(radiusmeanmantlist/R_earth,S50mantlist/1.e3,color='tab:green')
    ax[1,1].plot(radiusmeanmantlist/R_earth,S75mantlist/1.e3,color='tab:brown')
    ax[1,1].plot(radiusmeanmantlist/R_earth,S25meanmantlist/1.e3,color='y')
    ax[1,1].plot(radiusmeanmantlist/R_earth,Smeanmantlist/1.e3,color='g')
    ax[1,1].plot(radiusmeanmantlist/R_earth,S75meanmantlist/1.e3,color='brown')
    ax[1,1].set_xlabel(r"R $[R_\oplus]$")
    ax[1,1].set_ylabel('Sp. entropy S [kJ/K/kg]')
    ax[1,1].tick_params(axis='both', which='both', labelsize=18)
    
    ax[1,2].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,tmp_T[index][dataindexsort][indcoreEqPlanet],s=2,color=corecolor)
    ax[1,2].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,tmp_T[index][dataindexsort][indmantEqPlanet],s=2,color=mantlecolor)
    ax[1,2].plot(radiusmeancorelist[:-1]/R_earth,T25corelist[:-1],color='tab:cyan')
    ax[1,2].plot(radiusmeancorelist[:-1]/R_earth,T50corelist[:-1],color='tab:blue')
    ax[1,2].plot(radiusmeancorelist[:-1]/R_earth,T75corelist[:-1],color='tab:purple')
    ax[1,2].plot(radiusmeancorelist[:-1]/R_earth,T25meancorelist[:-1],color='c')
    ax[1,2].plot(radiusmeancorelist[:-1]/R_earth,Tmeancorelist[:-1],color='b')
    ax[1,2].plot(radiusmeancorelist[:-1]/R_earth,T75meancorelist[:-1],color='m')
    ax[1,2].plot(radiusmeanmantlist/R_earth,T25mantlist,color='tab:olive')
    ax[1,2].plot(radiusmeanmantlist/R_earth,T50mantlist,color='tab:green')
    ax[1,2].plot(radiusmeanmantlist/R_earth,T75mantlist,color='tab:brown')
    ax[1,2].plot(radiusmeanmantlist/R_earth,T25meanmantlist,color='y')
    ax[1,2].plot(radiusmeanmantlist/R_earth,Tmeanmantlist,color='g')
    ax[1,2].plot(radiusmeanmantlist/R_earth,T75meanmantlist,color='brown')
    ax[1,2].plot(radiusmeanlist/R_earth,solvusfn(np.abs(P25fitlist)),lw=2,color='red',label='W&M Fe-MgO Solvus')
    #ax[1,2].plot(radiusmeanlist/R_earth,vT_rubie(vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth)),color='black')
    ax[1,2].legend(fontsize=20)
    ax[1,2].set_xlabel(r"R $[R_\oplus]$")
    ax[1,2].set_ylabel('Temperature T [K]')
    ax[1,2].tick_params(axis='both', which='both', labelsize=18)
    
    ax[2,0].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,(u[index][dataindexsort][indcoreEqPlanet]-u0[indexin][data0inpartsort][indcoreEqPlanet])/1.e6,s=2,color=corecolor)
    ax[2,0].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,(u[index][dataindexsort][indmantEqPlanet]-u0[indexin][data0inpartsort][indmantEqPlanet])/1.e6,s=2,color=mantlecolor)
    ax[2,0].plot(radiusmeancorelist[:-1]/R_earth,deltau25corelist[:-1]/1.e6,color='tab:cyan')
    ax[2,0].plot(radiusmeancorelist[:-1]/R_earth,deltau50corelist[:-1]/1.e6,color='tab:blue')
    ax[2,0].plot(radiusmeancorelist[:-1]/R_earth,deltau75corelist[:-1]/1.e6,color='tab:purple')
    ax[2,0].plot(radiusmeancorelist[:-1]/R_earth,deltau25meancorelist[:-1]/1.e6,color='c')
    ax[2,0].plot(radiusmeancorelist[:-1]/R_earth,deltaumeancorelist[:-1]/1.e6,color='b')
    ax[2,0].plot(radiusmeancorelist[:-1]/R_earth,deltau75meancorelist[:-1]/1.e6,color='m')
    ax[2,0].plot(radiusmeanmantlist/R_earth,deltau25mantlist/1.e6,color='tab:olive')
    ax[2,0].plot(radiusmeanmantlist/R_earth,deltau50mantlist/1.e6,color='tab:green')
    ax[2,0].plot(radiusmeanmantlist/R_earth,deltau75mantlist/1.e6,color='tab:brown')
    ax[2,0].plot(radiusmeanmantlist/R_earth,deltau25meanmantlist/1.e6,color='y')
    ax[2,0].plot(radiusmeanmantlist/R_earth,deltaumeanmantlist/1.e6,color='g')
    ax[2,0].plot(radiusmeanmantlist/R_earth,deltau75meanmantlist/1.e6,color='brown')
    ax[2,0].set_xlabel(r"R $[R_\oplus]$")
    ax[2,0].set_ylabel('Sp. Energy change $\Delta U$ [MJ/kg]')
    ax[2,0].tick_params(axis='both', which='both', labelsize=18)
    
    ax[2,1].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,(tmp_s[index][dataindexsort][indcoreEqPlanet]-tmp_s0[indexin][data0inpartsort][indcoreEqPlanet])/1.e3,s=2,color=corecolor)
    ax[2,1].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,(tmp_s[index][dataindexsort][indmantEqPlanet]-tmp_s0[indexin][data0inpartsort][indmantEqPlanet])/1.e3,s=2,color=mantlecolor)
    ax[2,1].plot(radiusmeancorelist[:-1]/R_earth,deltaS25corelist[:-1]/1.e3,color='tab:cyan')
    ax[2,1].plot(radiusmeancorelist[:-1]/R_earth,deltaS50corelist[:-1]/1.e3,color='tab:blue')
    ax[2,1].plot(radiusmeancorelist[:-1]/R_earth,deltaS75corelist[:-1]/1.e3,color='tab:purple')
    ax[2,1].plot(radiusmeancorelist[:-1]/R_earth,deltaS25meancorelist[:-1]/1.e3,color='c')
    ax[2,1].plot(radiusmeancorelist[:-1]/R_earth,deltaSmeancorelist[:-1]/1.e3,color='b')
    ax[2,1].plot(radiusmeancorelist[:-1]/R_earth,deltaS75meancorelist[:-1]/1.e3,color='m')
    ax[2,1].plot(radiusmeanmantlist/R_earth,deltaS25mantlist/1.e3,color='tab:olive')
    ax[2,1].plot(radiusmeanmantlist/R_earth,deltaS50mantlist/1.e3,color='tab:green')
    ax[2,1].plot(radiusmeanmantlist/R_earth,deltaS75mantlist/1.e3,color='tab:brown')
    ax[2,1].plot(radiusmeanmantlist/R_earth,deltaS25meanmantlist/1.e3,color='y')
    ax[2,1].plot(radiusmeanmantlist/R_earth,deltaSmeanmantlist/1.e3,color='g')
    ax[2,1].plot(radiusmeanmantlist/R_earth,deltaS75meanmantlist/1.e3,color='brown')
    ax[2,1].set_xlabel(r"R $[R_\oplus]$")
    ax[2,1].set_ylabel('Sp. Entropy change $\Delta S$ [kJ/K/kg]')
    ax[2,1].tick_params(axis='both', which='both', labelsize=18)
    
    ax[2,2].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,(tmp_T[index][dataindexsort][indcoreEqPlanet]-tmp_T0[indexin][data0inpartsort][indcoreEqPlanet]),s=2,color=corecolor)
    ax[2,2].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,(tmp_T[index][dataindexsort][indmantEqPlanet]-tmp_T0[indexin][data0inpartsort][indmantEqPlanet]),s=2,color=mantlecolor)
    ax[2,2].plot(radiusmeancorelist[:-1]/R_earth,deltaT25corelist[:-1],color='tab:cyan')
    ax[2,2].plot(radiusmeancorelist[:-1]/R_earth,deltaT50corelist[:-1],color='tab:blue')
    ax[2,2].plot(radiusmeancorelist[:-1]/R_earth,deltaT75corelist[:-1],color='tab:purple')
    ax[2,2].plot(radiusmeancorelist[:-1]/R_earth,deltaT25meancorelist[:-1],color='c')
    ax[2,2].plot(radiusmeancorelist[:-1]/R_earth,deltaTmeancorelist[:-1],color='b')
    ax[2,2].plot(radiusmeancorelist[:-1]/R_earth,deltaT75meancorelist[:-1],color='m')
    ax[2,2].plot(radiusmeanmantlist/R_earth,deltaT25mantlist,color='tab:olive')
    ax[2,2].plot(radiusmeanmantlist/R_earth,deltaT50mantlist,color='tab:green')
    ax[2,2].plot(radiusmeanmantlist/R_earth,deltaT75mantlist,color='tab:brown')
    ax[2,2].plot(radiusmeanmantlist/R_earth,deltaT25meanmantlist,color='y')
    ax[2,2].plot(radiusmeanmantlist/R_earth,deltaTmeanmantlist,color='g')
    ax[2,2].plot(radiusmeanmantlist/R_earth,deltaT75meanmantlist,color='brown')
    ax[2,2].set_xlabel(r"R $[R_\oplus]$")
    ax[2,2].set_ylabel('Temperature change $\Delta T$ [K]')
    ax[2,2].tick_params(axis='both', which='both', labelsize=18)
    
    plt.tight_layout()
    imname=basename+'thermal_profiles.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    fig, ax = plt.subplots(1, 1, figsize=(4,4))
    ax.scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,tmp_T[index][dataindexsort][indcoreEqPlanet],s=2,color=corecolor)
    ax.scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,tmp_T[index][dataindexsort][indmantEqPlanet],s=2,color=mantlecolor)
    ax.fill_between(radiusmeancorelist[:-1]/R_earth,T25corelist[:-1],T75corelist[:-1],color='white',alpha=0.7)
    ax.plot(radiusmeancorelist[:-1]/R_earth,T25corelist[:-1],color='black',lw=1)#'tab:cyan'
    ax.plot(radiusmeancorelist[:-1]/R_earth,T50corelist[:-1],color='black',ls='--',lw=1)#'tab:blue'
    ax.plot(radiusmeancorelist[:-1]/R_earth,T75corelist[:-1],color='black',lw=1)#'tab:purple'
    #ax.plot(radiusmeancorelist[:-1]/R_earth,T25meancorelist[:-1],color='c')
    #ax.plot(radiusmeancorelist[:-1]/R_earth,Tmeancorelist[:-1],color='b')
    #ax.plot(radiusmeancorelist[:-1]/R_earth,T75meancorelist[:-1],color='m')
    ax.fill_between(radiusmeanmantlist[:-1]/R_earth,T25mantlist[:-1],T75mantlist[:-1],color='white',alpha=0.7)
    ax.plot(radiusmeanmantlist/R_earth,T25mantlist,color='black',lw=1.)#'tab:olive'
    ax.plot(radiusmeanmantlist/R_earth,T50mantlist,color='black',ls='--',lw=1.)#'tab:green'
    ax.plot(radiusmeanmantlist/R_earth,T75mantlist,color='black',lw=1.)#'tab:brown'
    #ax.plot(radiusmeanmantlist/R_earth,T25meanmantlist,color='y')
    #ax.plot(radiusmeanmantlist/R_earth,Tmeanmantlist,color='g')
    #ax.plot(radiusmeanmantlist/R_earth,T75meanmantlist,color='brown')
    ax.plot(radiusmeanlist/R_earth,solvusfn(np.abs(P25fitlist)),lw=2,color='blue',label='W&M Fe-MgO Solvus')
    #ax[1,2].plot(radiusmeanlist/R_earth,vT_rubie(vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth)),color='black')
    #ax.legend(fontsize=14)
    ax.set_xlabel(r"R $[R_\oplus]$")
    ax.set_ylabel('Temperature T [K]')
    ax.tick_params(axis='both', which='both', labelsize=12)
    plt.tight_layout()
    imname=basename+'Fig_12.pdf'
    plt.savefig(imname, dpi=100)
    plt.show()
    
    fig, ax = plt.subplots(2, 2, figsize=(18,12))
    ax[0,0].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,P[index][dataindexsort][indcoreEqPlanet]/1.e9,s=2,color=corecolor,label='Core particles')
    ax[0,0].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,P[index][dataindexsort][indmantEqPlanet]/1.e9,s=2,color=mantlecolor,label='Mantle particles')
    ax[0,0].plot(radiusmeanlist/R_earth,P25list/1.e9,label='25th %ile SPH value')
    #ax[0,0].plot(radiusmeanlist/R_earth,P25meanlist/1.e9)
    #ax[0,0].plot(radiusmeanlist/R_earth,P50list/1.e9)
    ax[0,0].plot(radiusmeanlist/R_earth,Pmeanlist/1.e9,label='Mean SPH value')
    ax[0,0].plot(radiusmeanlist/R_earth,P75list/1.e9,label='75th %ile SPH value')
    #ax[0,0].plot(radiusmeanlist/R_earth,P75meanlist/1.e9)
    ax[0,0].plot(radiusmeanlist/R_earth,vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,color='black',label='Static model',lw=3)
    if os.path.isfile(basename+'final'):
        ax[0,0].plot(HERC_r/R_earth,HERC_P/1.e9,'--',color='black',label='Rotating model',lw=3)
    ax[0,0].scatter(cmb_rubie/R_earth,P_rubie(cmb_rubie,rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,color='black',s=60,label='Static model CMB')
    if os.path.isfile(basename+'final'):
        ax[0,0].scatter(HERCc_layer.a/R_earth,HERC_PCMB/1.e9,color='black',s=45)
    #ax[0,0].plot(radiusmeanmantlist[:argmantfitmaxP25]/R_earth,P25mantfitfn(radiusmeanmantlist[:argmantfitmaxP25])/1.e9,lw=2)
    #ax[0,0].plot(radiusmeanlist[P25extendfit]/R_earth,P25extendfn(radiusmeanlist[P25extendfit])/1.e9,lw=2)
    #ax[0,0].plot(radiusmeancorelist[np.where(radiusmeancorelist<radiusmeanmantlist[0])]/R_earth,P25corefitfn(radiusmeancorelist[np.where(radiusmeancorelist<radiusmeanmantlist[0])])/1.e9,lw=2)
    #ax[0,0].plot(radiusmeanlist[P25extendfit]/R_earth,P25extendcorefn(radiusmeanlist[P25extendfit])/1.e9,lw=2,color='tab:purple')
    #ax[0,0].plot(radiusmeanlist/R_earth,P25fitlist/1.e9,'--',lw=1)
    #ax[0,0].plot(radiusmeanlist/R_earth,P50fitlist/1.e9,'--',lw=1)
    #ax[0,0].plot(radiusmeanlist/R_earth,P75fitlist/1.e9,'--',lw=1,color='lime')
    ax[0,0].set_xlabel(r"R $[R_\oplus]$")
    ax[0,0].set_ylabel('Pressure P [GPa]')
    ax[0,0].tick_params(axis='both', which='both', labelsize=20)
    ax[0,0].legend(fontsize=18)
    
    ax[0,1].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,tmp_T[index][dataindexsort][indcoreEqPlanet],s=2,color=corecolor,label='Core particles')
    ax[0,1].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,tmp_T[index][dataindexsort][indmantEqPlanet],s=2,color=mantlecolor,label='Mantle particles')
    ax[0,1].plot(radiusmeancorelist[:-1]/R_earth,T25corelist[:-1],color='tab:cyan',label='25th %ile SPH core')
    #ax[0,1].plot(radiusmeancorelist[:-1]/R_earth,T50corelist[:-1],color='tab:blue')
    ax[0,1].plot(radiusmeancorelist[:-1]/R_earth,T75corelist[:-1],color='tab:purple',label='75th %ile SPH core')
    #ax[0,1].plot(radiusmeancorelist[:-1]/R_earth,T25meancorelist[:-1],color='c')
    ax[0,1].plot(radiusmeancorelist[:-1]/R_earth,Tmeancorelist[:-1],color='b',label='Mean SPH core')
    #ax[0,1].plot(radiusmeancorelist[:-1]/R_earth,T75meancorelist[:-1],color='m')
    ax[0,1].plot(radiusmeanmantlist/R_earth,T25mantlist,color='tab:olive',label='25th %ile SPH mantle')
    #ax[0,1].plot(radiusmeanmantlist/R_earth,T50mantlist,color='tab:green')
    ax[0,1].plot(radiusmeanmantlist/R_earth,T75mantlist,color='tab:brown',label='75th %ile SPH mantle')
    #ax[0,1].plot(radiusmeanmantlist/R_earth,T25meanmantlist,color='y')
    ax[0,1].plot(radiusmeanmantlist/R_earth,Tmeanmantlist,color='g',label='Mean SPH mantle')
    #ax[0,1].plot(radiusmeanmantlist/R_earth,T75meanmantlist,color='brown')
    ax[0,1].plot(radiusmeanlist/R_earth,vT_rubie(vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)),color='black',label='Rubie (2015) model',lw=3)
    ax[0,1].scatter(cmb_rubie/R_earth,T_rubie(P_rubie(cmb_rubie,rCMB, rplanet, mtot/M_earth,mcore,mmant)),color='black',s=60,label='T&S model CMB')
    ax[0,1].set_xlabel(r"R $[R_\oplus]$")
    ax[0,1].set_ylabel('Temperature T [K]')
    ax[0,1].tick_params(axis='both', which='both', labelsize=20)
    ax[0,1].legend(fontsize=18)
    
    ax[1,0].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,P[index][dataindexsort][indcoreEqPlanet]/1.e9-vP_rubie(rrr[index][dataindexsort][indcoreEqPlanet], rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,s=2,color=corecolor,label='Core particles')
    ax[1,0].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,P[index][dataindexsort][indmantEqPlanet]/1.e9-vP_rubie(rrr[index][dataindexsort][indmantEqPlanet], rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,s=2,color=mantlecolor,label='Mantle particles')
    ax[1,0].plot(radiusmeanlist/R_earth,P25list/1.e9-vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,label='25th %ile SPH value')
    #ax[1,0].plot(radiusmeanlist/R_earth,P50list/1.e9)
    ax[1,0].plot(radiusmeanlist/R_earth,Pmeanlist/1.e9-vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,label='Mean SPH value')
    ax[1,0].plot(radiusmeanlist/R_earth,P75list/1.e9-vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,label='75th %ile SPH value')
    #ax[1,0].plot(radiusmeanlist/R_earth,P25meanlist/1.e9)
    #ax[1,0].plot(radiusmeanlist/R_earth,P75meanlist/1.e9)
    #ax[1,0].plot(radiusmeanlist/R_earth,vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth)/1.e9,color='black',label='T&S model')
    # ax[1,0].plot(radiusmeanmantlist[:argmantfitmaxP25]/R_earth,P25mantfitfn(radiusmeanmantlist[:argmantfitmaxP25])/1.e9-vP_rubie(radiusmeanmantlist[:argmantfitmaxP25], rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,lw=2,label='P25 polyfit')
    # ax[1,0].plot(radiusmeanlist[P25extendfit]/R_earth,P25extendfn(radiusmeanlist[P25extendfit])/1.e9-vP_rubie(radiusmeanlist[P25extendfit], rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,lw=2,label='P25 fit extension')
    # ax[1,0].plot(radiusmeancorelist[np.where(radiusmeancorelist<radiusmeanmantlist[0])]/R_earth,P25corefitfn(radiusmeancorelist[np.where(radiusmeancorelist<radiusmeanmantlist[0])])/1.e9-vP_rubie(radiusmeancorelist[np.where(radiusmeancorelist<radiusmeanmantlist[0])], rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,lw=2,label='P25 core polyfit')
    # ax[1,0].plot(radiusmeanlist[P25extendfit]/R_earth,P25extendcorefn(radiusmeanlist[P25extendfit])/1.e9-vP_rubie(radiusmeanlist[P25extendfit], rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,lw=2,label='P25 fit extension',color='tab:purple')
    # ax[1,0].plot(radiusmeanlist/R_earth,P25fitlist/1.e9-vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,'--',lw=1)
    # ax[1,0].plot(radiusmeanlist/R_earth,P50fitlist/1.e9-vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,'--',lw=1)
    # ax[1,0].plot(radiusmeanlist/R_earth,P75fitlist/1.e9-vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,'--',lw=1,color='lime')
    if os.path.isfile(basename+'final'):
        ax[1,0].plot(HERC_r/R_earth,HERC_P/1.e9-vP_rubie(HERC_r, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,'--',color='black',label='Spun HERCULES model',lw=3)
        ax[1,0].scatter(HERCc_layer.a/R_earth,HERC_PCMB/1.e9-vP_rubie(HERCc_layer.a, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,color='black',s=45)
    ax[1,0].axhline(y=0, color='k',lw=3)
    ax[1,0].scatter(cmb_rubie/R_earth,0,color='black',s=60,label='T&S model CMB')
    ax[1,0].set_xlabel(r"R $[R_\oplus]$")
    ax[1,0].set_ylabel('P deviation from model [GPa]')
    ax[1,0].tick_params(axis='both', which='both', labelsize=20)
    
    ax[1,1].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,tmp_T[index][dataindexsort][indcoreEqPlanet]-vT_rubie(vP_rubie(rrr[index][dataindexsort][indcoreEqPlanet], rCMB, rplanet, mtot/M_earth,mcore,mmant)),s=2,color=corecolor,label='Core particles')
    ax[1,1].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,tmp_T[index][dataindexsort][indmantEqPlanet]-vT_rubie(vP_rubie(rrr[index][dataindexsort][indmantEqPlanet], rCMB, rplanet, mtot/M_earth,mcore,mmant)),s=2,color=mantlecolor,label='Mantle particles')
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,T25corelist[:-1]-vT_rubie(vP_rubie(radiusmeancorelist[:-1], rCMB, rplanet, mtot/M_earth,mcore,mmant)),color='tab:cyan',label='25th %ile SPH core')
    #ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,T50corelist[:-1],color='tab:blue')
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,T75corelist[:-1]-vT_rubie(vP_rubie(radiusmeancorelist[:-1], rCMB, rplanet, mtot/M_earth,mcore,mmant)),color='tab:purple',label='75th %ile SPH core')
    #ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,T25meancorelist[:-1],color='c')
    ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,Tmeancorelist[:-1]-vT_rubie(vP_rubie(radiusmeancorelist[:-1], rCMB, rplanet, mtot/M_earth,mcore,mmant)),color='b',label='Mean SPH core')
    #ax[1,1].plot(radiusmeancorelist[:-1]/R_earth,T75meancorelist[:-1],color='m')
    ax[1,1].plot(radiusmeanmantlist/R_earth,T25mantlist-vT_rubie(vP_rubie(radiusmeanmantlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)),color='tab:olive',label='25th %ile SPH mantle')
    #ax[1,1].plot(radiusmeanmantlist/R_earth,T50mantlist,color='tab:green')
    ax[1,1].plot(radiusmeanmantlist/R_earth,T75mantlist-vT_rubie(vP_rubie(radiusmeanmantlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)),color='tab:brown',label='75th %ile SPH mantle')
    #ax[1,1].plot(radiusmeanmantlist/R_earth,T25meanmantlist,color='y')
    ax[1,1].plot(radiusmeanmantlist/R_earth,Tmeanmantlist-vT_rubie(vP_rubie(radiusmeanmantlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)),color='g',label='Mean SPH mantle')
    #ax[1,1].plot(radiusmeanmantlist/R_earth,T75meanmantlist,color='brown')
    #ax[1,1].plot(radiusmeanlist/R_earth,vT_rubie(vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth)),color='black',label='Rubie (2015) model')
    ax[1,1].axhline(y=0, color='k',lw=3)
    ax[1,1].scatter(cmb_rubie/R_earth,0,color='black',s=60,label='T&S model CMB')
    ax[1,1].set_xlabel(r"R $[R_\oplus]$")
    ax[1,1].set_ylabel('T deviation from model [K]')
    ax[1,1].tick_params(axis='both', which='both', labelsize=20)
    
    plt.tight_layout()
    imname=basename+'PT_comparison.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    fig, ax = plt.subplots(2, 1, figsize=(6,8))
    ax[0].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,P[index][dataindexsort][indcoreEqPlanet]/1.e9,s=2,color=corecolor,label='Core particles')
    ax[0].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,P[index][dataindexsort][indmantEqPlanet]/1.e9,s=2,color=mantlecolor,label='Mantle particles')
    ax[0].plot(radiusmeanlist/R_earth,vP_rubie(radiusmeanlist, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,color='black',label='Static model',lw=3)
    if os.path.isfile(basename+'final'):
        ax[0].plot(HERC_r/R_earth,HERC_P/1.e9,'--',color='black',label='Rotating model',lw=3)
    ax[0].scatter(cmb_rubie/R_earth,P_rubie(cmb_rubie,rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,color='black',s=45,label='Model CMB')
    if os.path.isfile(basename+'final'):
        ax[0].scatter(HERCc_layer.a/R_earth,HERC_PCMB/1.e9,color='black',s=45)
    ax[0].set_xlabel(r"R $[R_\oplus]$",size=14)
    ax[0].set_ylabel('Pressure P [GPa]',size=14)
    ax[0].tick_params(axis='both', which='both', labelsize=12)
    ax[0].legend(fontsize=14)
    
    ax[1].scatter(rrr[index][dataindexsort][indcoreEqPlanet]/R_earth,P[index][dataindexsort][indcoreEqPlanet]/1.e9-vP_rubie(rrr[index][dataindexsort][indcoreEqPlanet], rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,s=2,color=corecolor,label='Core particles')
    ax[1].scatter(rrr[index][dataindexsort][indmantEqPlanet]/R_earth,P[index][dataindexsort][indmantEqPlanet]/1.e9-vP_rubie(rrr[index][dataindexsort][indmantEqPlanet], rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,s=2,color=mantlecolor,label='Mantle particles')
    if os.path.isfile(basename+'final'):
        ax[1].plot(HERC_r/R_earth,HERC_P/1.e9-vP_rubie(HERC_r, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,'--',color='black',label='Spun HERCULES model',lw=3)
        ax[1].scatter(HERCc_layer.a/R_earth,HERC_PCMB/1.e9-vP_rubie(HERCc_layer.a, rCMB, rplanet, mtot/M_earth,mcore,mmant)/1.e9,color='black',s=45)
    ax[1].axhline(y=0, color='k',lw=3)
    ax[1].scatter(cmb_rubie/R_earth,0,color='black',s=45,label='T&S model CMB')
    ax[1].set_xlabel(r"R $[R_\oplus]$",size=14)
    ax[1].set_ylabel('P deviation from model [GPa]',size=14)
    ax[1].tick_params(axis='both', which='both', labelsize=12)
    
    plt.tight_layout()
    imname=basename+'Fig_11.pdf'
    plt.savefig(imname, dpi=100)
    plt.show()

# ----------------------------- # 
## gracefully quit if we're calling this code from the jupyter notebook - don't need to run through the whole txt file
if __name__=='__main__':
    
    dir = "C:/Users/gerri/swift-tutorial/survey data/mass1to2/halfearth/halfearth0.5x0.25th30/"
    listfilename = 'compare_halfearth0.5x0.25th30.txt'
    listfilename = dir+listfilename
    runlist = np.loadtxt(listfilename,skiprows=1,delimiter='|',dtype={'names':('basename','numsnap','label','write'),'formats':('U256',int,'U256',int)})
    
    for i in range(np.size(runlist)):
        basename = runlist[i]['basename']
        Nsnap = runlist[i]['numsnap']-1
        
        snapshot0 = basename+'_%04d.hdf5' % 0
        data0 = sw.load(snapshot0)
        data0.create_particle_datasets()
        data0.gas.coordinates.convert_to_mks()
        data0.gas.pressures.convert_to_mks()
        data0.gas.internal_energies.convert_to_mks()
        data0.gas.masses.convert_to_mks()
        data0.gas.velocities.convert_to_mks()
        data0.gas.smoothing_lengths.convert_to_mks()
        data0.gas.densities.convert_to_mks()
        data0.gas.potentials.convert_to_mks()
        
        Ntot = data0.gas.metadata.n_gas
        
        cm,vcm = find_initial_com(data0)
        print('Initial CoM/vCoM: ',cm,vcm)
    
# Nbound = np.empty([Nsnap])
# Nplanet = np.empty([Nsnap])
# Ndisk = np.empty([Nsnap])
# Mbound = np.empty([Nsnap])
# Mplanet = np.empty([Nsnap])
# Mdisk = np.empty([Nsnap])
# KEtot = np.empty([Nsnap])
# IEtot = np.empty([Nsnap])
# GPEtot = np.empty([Nsnap])
# KEbnd= np.empty([Nsnap])
# IEbnd = np.empty([Nsnap])
# GPEbnd = np.empty([Nsnap])
# KEplanet= np.empty([Nsnap])
# IEplanet = np.empty([Nsnap])
# GPEplanet = np.empty([Nsnap])
# KEdisk= np.empty([Nsnap])
# IEdisk = np.empty([Nsnap])
# GPEdisk = np.empty([Nsnap])
# GPEmin = 0
        time = np.empty([Nsnap])
        
        if os.path.isfile(basename+'_%04d.hdf5'%Nsnap):
            snapshot = basename+'_%04d.hdf5' % Nsnap
        else:
            snapshot = basename+'_%d.hdf5' % Nsnap
        print('Reading snapshot: ',snapshot)
        data = sw.load(snapshot)
        data.create_particle_datasets()
        data.gas.coordinates.convert_to_mks()
        data.gas.pressures.convert_to_mks()
        data.gas.internal_energies.convert_to_mks()
        data.gas.masses.convert_to_mks()
        data.gas.velocities.convert_to_mks()
        data.gas.smoothing_lengths.convert_to_mks()
        data.gas.densities.convert_to_mks()
        data.gas.potentials.convert_to_mks()    
        snaptime = data.metadata.time.in_mks()
        
        indbnd, Mbound,pos,vel = bound_mass(data, cm, vcm)
        Nbound = np.shape(indbnd)[1]
        databnd = apply_index(data, indbnd)
        bcm,bvcm = find_initial_com(databnd)
        inddisk, indplanet, diskm, planetm, rplanet = find_disk(databnd, bcm, bvcm, pos[indbnd], vel[indbnd],basename=basename)
        #trying again for double-checking in accretionary cases
        inddisk2, indplanet2, diskm2, planetm2, rplanet2 = find_disk(databnd, bcm, bvcm, pos[indbnd], vel[indbnd],basename=basename,rplanetmax=rplanet*R_earth)
        if (planetm/(planetm+diskm)-planetm2/(planetm2+diskm2))<0.025:
            inddisk, indplanet, diskm, planetm, rplanet = inddisk2, indplanet2, diskm2, planetm2, rplanet2
            print('Planet/disk boundary: r = ',rplanet,'R_earth')
            print('Planet mass: ',planetm,' fraction of bound mass: ',planetm/(planetm+diskm))
            print('Disk mass: ',diskm,' fraction of bound mass: ',diskm/(planetm+diskm))
            
        print('Bound mass CoM/vCoM: ',bcm,bvcm)
        
        plot_IE_gain(data0, data, bcm, bvcm, pos, vel, index=indbnd, rplanet=rplanet, basename=basename)
        if (os.path.isfile(basename+'_profiles.txt')==False):
            plot_IE_gain(data0, data, bcm, bvcm, pos, vel, index=indbnd, rplanet=rplanet, basename=basename)
        if (os.path.getmtime(snapshot)>os.path.getmtime(basename+'_profiles.txt')):
            plot_IE_gain(data0, data, bcm, bvcm, pos, vel, index=indbnd, rplanet=rplanet, basename=basename)
        if (os.path.isfile(basename+'_isentropes.txt')==False):
            isentrope_profiles(data, bcm, bvcm, pos, vel, index=indbnd, basename=basename, rplanet=rplanet)
        if (os.path.getmtime(snapshot)>os.path.getmtime(basename+'_isentropes.txt')):
            isentrope_profiles(data, bcm, bvcm, pos, vel, index=indbnd, basename=basename, rplanet=rplanet)
        if (os.path.isfile(basename+'_miscibility.txt')==False):
            plot_miscibility(data, bcm, bvcm, pos, vel, index=indbnd, basename=basename, rplanet=rplanet)
        if (os.path.getmtime(snapshot)>os.path.getmtime(basename+'_miscibility.txt')):
            plot_miscibility(data, bcm, bvcm, pos, vel, index=indbnd, basename=basename, rplanet=rplanet)
        if (os.path.isfile(basename+'_L_profiles.txt')==False):
            Lz_profiles(data, bcm, bvcm, pos, vel, index=indbnd)
        if (os.path.getmtime(snapshot)>os.path.getmtime(basename+'_L_profiles.txt')):
            Lz_profiles(data, bcm, bvcm, pos, vel, index=indbnd)
        #print(indplanet)
        if (os.path.isfile(basename+'_input.yml')==False):
            HERCULES_profiles(databnd, bcm, bvcm, pos[indbnd], vel[indbnd], index=indplanet, basename=basename)
        if (os.path.getmtime(snapshot)>os.path.getmtime(basename+'_input.yml')):
            HERCULES_profiles(databnd, bcm, bvcm, pos[indbnd], vel[indbnd], index=indplanet, basename=basename)

# for snapshot_id in range(0,Nsnap): #USER INPUT (snapshot index + 1)
#     #deltasnap=100 # USER INPUT time between each snapshot 
#     snapshot = basename+'_%04d.hdf5' % snapshot_id
    
#     data = sw.load(snapshot)
#     data.gas.coordinates.convert_to_mks()
#     data.gas.pressures.convert_to_mks()
#     data.gas.internal_energies.convert_to_mks()
#     data.gas.masses.convert_to_mks()
#     data.gas.velocities.convert_to_mks()
#     data.gas.smoothing_lengths.convert_to_mks()
#     data.gas.densities.convert_to_mks()
#     data.gas.potentials.convert_to_mks()    
#     snaptime = data.metadata.time.in_mks()
    
#     print('Simulation time: ',snaptime/3600,' hours')
    
#     KEtot[snapshot_id], IEtot[snapshot_id], GPEtot[snapshot_id], newGPEmin = find_energy(data, cm, vcm)
#     GPEmin = np.amin([GPEmin,newGPEmin])
#     print('GPEmin: ',GPEmin)
    
#     indbnd, Mbound[snapshot_id] = bound_mass(data, cm, vcm)
#     Nbound[snapshot_id] = np.shape(indbnd)[1]
#     databnd = apply_index(data, indbnd)
#     bcm,bvcm = find_initial_com(databnd)
#     print('Bound mass CoM/vCoM: ',bcm,bvcm)
    
#     inddisk, indplanet, Mdisk[snapshot_id], Mplanet[snapshot_id] = find_disk(databnd, cm, vcm)
#     datadisk = apply_index(databnd, inddisk)
#     Ndisk[snapshot_id] = np.shape(inddisk)[1]
#     dataplanet = apply_index(databnd, indplanet)
#     Nplanet[snapshot_id] = np.shape(indplanet)[1]
    
#     KEbnd[snapshot_id], IEbnd[snapshot_id], GPEbnd[snapshot_id], temp = find_energy(databnd, cm, vcm)
#     KEdisk[snapshot_id], IEdisk[snapshot_id], GPEdisk[snapshot_id], temp = find_energy(datadisk, cm, vcm)
#     KEplanet[snapshot_id], IEplanet[snapshot_id], GPEplanet[snapshot_id], temp = find_energy(dataplanet, cm, vcm)
    
#PGPEtot = GPEtot - GPEmin*Ntot
#PGPEbnd = GPEbnd - GPEmin*Nbound
#PGPEdisk = GPEdisk - GPEmin*Ndisk
#PGPEplanet = GPEplanet - GPEmin*Nplanet
#Etot = PGPEtot + KEtot + IEtot
#Ebnd = PGPEbnd + KEbnd + IEbnd
#Edisk = PGPEdisk + KEdisk + IEdisk
#Eplanet = PGPEplanet + KEplanet + IEplanet
#Etot0 = Etot[0]

#fir2, ax = plt.subplots(1, 1, figsize=(10,6))
#ax.scatter(,label='Lz, initial Lz=%04f LEM'%Lztot0)
#ax.scatter(,label='Lw0, initial Lw0=%04f LEM'%Lw0tot0)
#ax.set_xlabel('time (hours)')
#ax.set_ylabel('Change in total angular momentum (LEM)')
#ax.axis('equal')
#ax.set_ylim(2.5,3.5)
#ax.legend()