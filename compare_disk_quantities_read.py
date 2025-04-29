# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:48:34 2022

@author: adriana
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import unyt
from numba import njit
import h5py
import struct

import os,sys

import swiftsimio as sw
import woma
from woma.misc import utils, io
from woma.eos import tillotson, sesame, idg, hm80
from woma.eos.T_rho import T_rho
from woma.misc import glob_vars as gv
from copy import deepcopy
from scipy import interpolate

#this_dir, this_file = os.path.split(__file__)
#path = os.path.join(this_dir)
#sys.path.append(path)
#import gadget_sph

R_earth = 6.371e6   # m
M_earth = 5.9724e24  # kg 
G = 6.67408e-11  # m^3 kg^-1 s^-2
LEM=3.5E34 #AMof Earth-Moon system in mks

mant_mat_id = [400,403] #USER INPUT
core_mat_id = [401,402] #USER INPUT

solvusfn = interpolate.interp1d([0,50.e9,100.e9,400.e9,1000.e9],[4080,6094,6752,9337,14507.0])

def find_initial_com(data,index=None):
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
    if index is None:
        index = np.where(data.gas.particle_ids.value>=0)
    mat_id = data.gas.material_ids.value[index]
    part_id = data.gas.particle_ids.value[index]
    pos = data.gas.coordinates[index]
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth
    vel = data.gas.velocities[index]
    vel_x = vel.value[:,0]
    vel_y = vel.value[:,1]
    vel_z = vel.value[:,2]
    m = data.gas.masses.value[index]
    pot = data.gas.potentials.value[index]
    
    mtot = np.sum(m,dtype='float64')
    xcm = np.sum(pos_x*m,dtype='float64')/mtot
    ycm = np.sum(pos_y*m,dtype='float64')/mtot
    zcm = np.sum(pos_z*m,dtype='float64')/mtot
    cm = [xcm,ycm,zcm] #not including 40 R_earth offset!
    vxcm = np.sum(vel_x*m,dtype='float64')/mtot
    vycm = np.sum(vel_y*m,dtype='float64')/mtot
    vzcm = np.sum(vel_z*m,dtype='float64')/mtot
    vcm = [vxcm,vycm,vzcm] 
    
    return cm,vcm

def find_w(data,cm,vcm,index=None):
    '''
    Finds the omega vector for an input dataset

    Parameters
    ----------
    data : swift dataset
    cm : initial center of mass offset - from find_initial_com
    vcm: initial velocity center of mass offset - from find_initial_com

    Returns
    -------
    w : [wx,wy,wz]
        Normed omega vector (length is one) - orientation of spin axis of AM

    '''
    if index is None:
        index = np.where(data.gas.particle_ids.value>=0)
    pos = data.gas.coordinates[index]
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    vel = data.gas.velocities[index]
    vel_x = vel.value[:,0] - vcm[0]
    vel_y = vel.value[:,1] - vcm[1]
    vel_z = vel.value[:,2] - vcm[2]
    m = data.gas.masses.value[index]
    mtot = np.sum(m,dtype='float64')
    
    ryz2 = np.power(pos_y,2.0) + np.power(pos_z,2.0)
    rxz2 = np.power(pos_x,2.0) + np.power(pos_z,2.0)
    rxy2 = np.power(pos_x,2.0) + np.power(pos_y,2.0)
    wx = np.sum(m*np.divide(pos_y*vel_z - pos_z*vel_y, ryz2),dtype='float64') / mtot #Ntot
    wy = np.sum(m*np.divide(pos_z*vel_x - pos_x*vel_z, rxz2),dtype='float64') / mtot #Ntot
    wz = np.sum(m*np.divide(pos_x*vel_z - pos_z*vel_x, rxy2),dtype='float64') / mtot #Ntot
    wlen = np.sqrt(wx**2 + wy**2 + wz**2)
    
    w = [wx,wy,wz]/wlen
    return w

def find_Lz(data,cm,vcm,index=None):
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
    pos = data.gas.coordinates[index]
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    #pos_z = pos.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    vel = data.gas.velocities[index]
    vel_x = vel.value[:,0] - vcm[0]
    vel_y = vel.value[:,1] - vcm[1]
    #vel_z = vel.value[:,2] - vcm[2]
    m = data.gas.masses.value[index]
    
    Lz = np.abs(np.sum(m * (pos_x*vel_y - pos_y*vel_x),dtype='float64'))
    return Lz/LEM

def find_Lw(data,cm,vcm,w,index=None):
    '''
    Finds angular momentum for an input dataset around a given w-axis

    Parameters
    ----------
    data : swift dataset
    cm : center of mass offset - from find_initial_com
    vcm: velocity center of mass offset - from find_initial_com
    w : [wx,wy,wz]
        Normed omega vector (length is one) - orientation of spin axis of AM

    Returns
    -------
    Lw : float64
        Total angular momentum for input dataset - in LEM units

    '''
    if index is None:
        index = np.where(data.gas.particle_ids.value>=0)
    wx,wy,wz = w
    pos = data.gas.coordinates[index]
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    vel = data.gas.velocities[index]
    vel_x = vel.value[:,0] - vcm[0]
    vel_y = vel.value[:,1] - vcm[1]
    vel_z = vel.value[:,2] - vcm[2]
    m = data.gas.masses.value[index]
    
    Lw = np.abs(np.sum(m * (vel_x*(wy*pos_z - wz*pos_y) - vel_y*(wx*pos_z - wz*pos_x) + vel_z*(wx*pos_y - wy*pos_x)),dtype='float64'))
    return Lw/LEM

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
    mask.constrain_spatial([None,None,None])
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
    
    indmantle = np.where(np.isin(mat_id,mant_mat_id))
    indcore = np.where(np.isin(mat_id,core_mat_id))
    
    coreSort = np.argsort(pot[indcore])
    xcm = np.mean(pos_x[indcore][coreSort[0:Ncenter]])
    ycm = np.mean(pos_y[indcore][coreSort[0:Ncenter]])
    zcm = np.mean(pos_z[indcore][coreSort[0:Ncenter]])
    vxcm = np.mean(vel_x[indcore][coreSort[0:Ncenter]])
    vycm = np.mean(vel_y[indcore][coreSort[0:Ncenter]])
    vzcm = np.mean(vel_z[indcore][coreSort[0:Ncenter]])
    bndm = np.sum(m[indcore][coreSort],dtype='float64')
    oldBoundPot = np.mean(pot[indcore][coreSort[0:Ncenter]])
    print('center: ', xcm/R_earth, ycm/R_earth, zcm/R_earth)
    
    #iterating - still not entirely confident about what's going on here - update: maybe more confident
    #testing this with just bound core mass at the moment? might see how things differ
    indbnd = np.where(m<0) #setting to return an empty index in the case where nothing is bound
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
        newm = np.sum(m[indbnd],dtype='float64')
        escm = np.sum(m[indesc],dtype='float64')

        indbndCore = np.where(np.isin(mat_id,core_mat_id) & (KE+PE < 0.0))
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
        
    return indbnd, newm/M_earth, pos2, vel2

def find_disk(data,cm,vcm,pos,vel,index=None):
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
    #bcm,bvcm = find_initial_com(data)
    #for i in range(3):
        #bcm[i]-=cm[i]
        #bvcm[i]-=vcm[i]
    #print('Bound CoM/vCoM: ',bcm,bvcm)
    if index is None:
        index = np.where(data.gas.particle_ids.value>=0)
    
    mat_id = data.gas.material_ids.value[index]
    part_id = data.gas.particle_ids.value[index]
    data.gas.coordinates.convert_to_mks()
    #pos = data.gas.coordinates
    boxsize = data.metadata.boxsize.value
    pos_x = pos[index].value[:,0]-boxsize[0]/2*R_earth #- cm[0]
    pos_y = pos[index].value[:,1]-boxsize[0]/2*R_earth #- cm[1]
    pos_z = pos[index].value[:,2]-boxsize[0]/2*R_earth #- cm[2]
    rrr = np.sqrt((pos_x)**2 + (pos_y)**2 + (pos_z)**2)
    #vel = data.gas.velocities
    vel_x = vel[index].value[:,0] #- vcm[0]
    vel_y = vel[index].value[:,1] #- vcm[1]
    vel_z = vel[index].value[:,2] #- vcm[2]
    v2 = np.power(vel_x,2.0) + np.power(vel_y,2.0) + np.power(vel_z,2.0)
    vvv = np.sqrt(v2)
    m = data.gas.masses.value[index]
    KE = 0.5 * m * v2
    pot = data.gas.potentials.value[index]
    Lz = m*(pos_x*vel_y - pos_y*vel_x)
    
    # Using a moving average calculation to find the planet-disk boundary (max particle KE)
    Rsort = np.argsort(rrr)
    Navg = 500
    if len(rrr)<Navg:
        try:
            rplanet = np.amax(rrr)
        except:
            return np.array([]), np.array([]), np.array([]), np.array([]), 0, 0, 0, 0
    else:
        KEmax = 0
        #KEmin = np.amax(KE)
        rKEmax = 0
        rKEmin = 0
        KEavglist = np.zeros(len(rrr)-Navg-1)
        KEdifflist = np.zeros(len(rrr)-Navg-2)
        #vvvavglist = np.zeros(len(rrr)-Navg-1)
        rcutoff = rrr[Rsort[-1]]
        kecutoff = rrr[Rsort[-1]]
        rfirstpeak = 0
        for i in range(len(rrr)-Navg-1):
            KEavg = np.sum(KE[Rsort[i:i+Navg]],dtype='float64')/Navg
            #vvvavg = np.sum(vvv[Rsort[i:i+Navg]])/Navg
            KEavglist[i]=KEavg
            if i>0: 
                KEdifflist[i-1]=KEavg - KEavglist[i-1]
            #vvvavglist[i] = vvvavg
            ravg = np.sum(rrr[Rsort[i+1:i+Navg]]-rrr[Rsort[i:i+Navg-1]],dtype='float64')/(Navg-1)
            KEdiff = np.sum(KE[Rsort[i+1:i+Navg]]-KE[Rsort[i:i+Navg-1]],dtype='float64')/(Navg-1)
            if KEavg > KEmax:
                KEmax = KEavg
                rKEmax = rrr[Rsort[i]]
                #KEmin = KEmax
            #else:
            #    if KEavg < KEmin:
            #        rKEmin = rrr[Rsort[i]]
            if (rrr[Rsort[i+Navg+1]]-rrr[Rsort[i+Navg]]) > 1000*ravg:
                rcutoff = np.amin([rrr[Rsort[i]],rcutoff])
            if (KE[Rsort[i+Navg+1]]-KE[Rsort[i+Navg]]) > 1000*KEdiff:
                kecutoff = np.amin([rrr[Rsort[i]],kecutoff])
        rdvpeak = rrr[Rsort[np.argmin(KEdifflist)+Navg-2]]
        irdvpeak = np.argmin(KEdifflist)+Navg-2
        nn = 0
        while (rdvpeak >  (np.std(rrr)*2)) & (nn<5):
            rdvpeak = rrr[Rsort[np.argmin(KEdifflist[:irdvpeak-Navg+2])]]
            irdvpeak = np.argmin(KEdifflist[:np.argmin(KEdifflist)])-Navg+2
            nn+=1
        iKEdvpeak = np.argmax(KEavglist[:irdvpeak])
        rKEdvpeak = rrr[Rsort[iKEdvpeak]]
        print ('cutoff values: ',rKEmax/R_earth,rcutoff/R_earth,rKEmin/R_earth,rdvpeak/R_earth,rKEdvpeak/R_earth)
        #if rcutoff > kecutoff:
            #rcutoff = kecutoff
        rplanet = np.amin([rKEmax,rKEdvpeak])
        if (rdvpeak < rKEmax < (1.03*rdvpeak)) & (rKEdvpeak < rdvpeak < (1.03*rKEdvpeak)): #sometimes the above misses outer layer of very defined planets
            rplanet = rKEmax
        meanmat = np.mean(mat_id[np.where(rrr<rplanet)])
        #print('Mean mat id: ',meanmat)
        #check to see if we accidentally ended up at the CMB - in this case planets are well-formed so we take overall KE peak
        if 400 in mat_id: mant_mat_id_temp = 400
        if 401 in mat_id: core_mat_id_temp = 401
        if 402 in mat_id: core_mat_id_temp = 402
        if 403 in mat_id: mant_mat_id_temp = 403
        if mant_mat_id_temp>core_mat_id_temp:
            if meanmat < (.95*core_mat_id_temp+.05*mant_mat_id_temp):
                rplanet = rKEmax
        if mant_mat_id_temp<core_mat_id_temp:
            if meanmat > (.95*core_mat_id_temp+.05*mant_mat_id_temp):
                rplanet = rKEmax
    inddisk = np.where(rrr > rplanet)
    indplanet = np.where(rrr <= rplanet)
    diskm = np.sum(m[inddisk],dtype='float64')
    planetm = np.sum(m[indplanet],dtype='float64')
    indmantle = np.where(np.isin(mat_id,mant_mat_id) & (rrr <= rplanet))
    indcore = np.where(np.isin(mat_id,core_mat_id) & (rrr <= rplanet))
    mantlem = np.sum(m[indmantle],dtype='float64')
    corem = np.sum(m[indcore],dtype='float64')
    
    print('Planet/disk boundary: r = ',rplanet/R_earth,'R_earth')
    print('Planet mass: ',planetm/M_earth,' fraction of bound mass: ',planetm/np.sum(m,dtype='float64'))
    print('Disk mass: ',diskm/M_earth,' fraction of bound mass: ',diskm/np.sum(m,dtype='float64'))
    print('Mantle mass: ',mantlem/M_earth,' fraction of bound mass: ',mantlem/np.sum(m,dtype='float64'))
    print('Core mass: ',corem/M_earth,' fraction of bound mass: ',corem/np.sum(m,dtype='float64'))
    
    return inddisk, indplanet, indmantle, indcore, diskm/M_earth, planetm/M_earth, mantlem/M_earth, corem/M_earth
    

def find_energy(data,cm,vcm,index=None):
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
    if index is None:
        index = np.where(data.gas.particle_ids.value>=0)
    pos = data.gas.coordinates[index]
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    vel = data.gas.velocities[index]
    vel_x = vel.value[:,0] - vcm[0]
    vel_y = vel.value[:,1] - vcm[1]
    vel_z = vel.value[:,2] - vcm[2]
    m = data.gas.masses.value[index]
    pot = data.gas.potentials.value[index]
    u = data.gas.internal_energies.value[index]
    
    v2 = np.power(vel_x,2.0) + np.power(vel_y,2.0) + np.power(vel_z,2.0)
    ke = 0.5*m*v2
    KEtot = np.sum(ke,dtype='float64')
    ie = u*m
    IEtot = np.sum(ie,dtype='float64')
    gpe = pot*m * 0.5 #correct for extra factor of two
    GPEtot = np.sum(gpe,dtype='float64')
    try:
        GPEmin = np.amin(gpe)
    except:
        GPEmin=0
    
    return KEtot,IEtot,GPEtot,GPEmin

def find_h_max_parts(data,index=None):
    '''
    Finds total number of particles at density floor

    Parameters
    ----------
    data : swift dataset

    Returns
    -------
    N_h_max

    '''
    if index is None:
        index = np.where(data.gas.particle_ids.value>=0)
    data.gas.smoothing_lengths.convert_to_mks()
    h = data.gas.smoothing_lengths.value[index]
    h_max=data.metadata.parameters.get('SPH:h_max').astype(float)*R_earth
    hmax_parts = np.where(h>=h_max*.999)
    return np.shape(hmax_parts)[1]

def find_lost_energy(data,cm,vcm,snapshot_id,basename):
    '''
    Finds energy of particles lost between snapshots - add correction to overall energy conservation

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    snapshot_id : TYPE
        DESCRIPTION.

    Returns
    -------
    KElost, IElost, GPElost
        KElost: total of last-seen kinetic energy values for lost particles (J)
        IElost: total of last-seen internal energy values for lost particles (J)
        GPE lost: total of last-seen gravitational potential energy values for lost particles (J)

    '''
    part_id = data.gas.particle_ids.value
    # pos = data.gas.coordinates
    # pos_x = pos.value[:,0]-40*R_earth - cm[0]
    # pos_y = pos.value[:,1]-40*R_earth - cm[1]
    # pos_z = pos.value[:,2]-40*R_earth - cm[2]
    # vel = data.gas.velocities
    # vel_x = vel.value[:,0] - vcm[0]
    # vel_y = vel.value[:,1] - vcm[1]
    # vel_z = vel.value[:,2] - vcm[2]
    # m = data.gas.masses.value
    # pot = data.gas.potentials.value
    # u = data.gas.internal_energies.value
    
    # v2 = np.power(vel_x,2.0) + np.power(vel_y,2.0) + np.power(vel_z,2.0)
    # ke = 0.5*m*v2
    # #KEtot = np.sum(ke)
    # ie = u*m
    # #IEtot = np.sum(ie)
    # gpe = pot*m * 0.5 #correct for extra factor of two
    # #GPEtot = np.sum(gpe)
    # #GPEmin = np.amin(gpe)
    
    snapshot0 = basename+'_%04d.hdf5' % (snapshot_id-1)
    try:
        data0 = sw.load(snapshot0)
        data0.gas.coordinates.convert_to_mks()
    except:
        print('Error loading gas data in previous file: ',snapshot)
        print('Last stable snapshot will be used')
        stable = False
        j=2
        while not stable:
            snapshot0 = basename+'_%04d.hdf5' % (snapshot_id-j)
            try:
                data0 = sw.load(snapshot0)
                data0.gas.coordinates.convert_to_mks()
            except:
                j+=1
            else:
                stable = True
                print('Last stable snapshot: ',snapshot0)
    #data0 = sw.load(snapshot0)
    #data0.gas.coordinates.convert_to_mks()
    part_id0 = data0.gas.particle_ids.value
    data0.gas.pressures.convert_to_mks()
    data0.gas.internal_energies.convert_to_mks()
    data0.gas.masses.convert_to_mks()
    data0.gas.velocities.convert_to_mks()
    data0.gas.smoothing_lengths.convert_to_mks()
    data0.gas.densities.convert_to_mks()
    data0.gas.potentials.convert_to_mks()
    
    pos0 = data0.gas.coordinates
    boxsize = data0.metadata.boxsize.value
    pos_x0 = pos0.value[:,0]-boxsize[0]/2*R_earth - cm[0]
    pos_y0 = pos0.value[:,1]-boxsize[1]/2*R_earth - cm[1]
    pos_z0 = pos0.value[:,2]-boxsize[2]/2*R_earth - cm[2]
    vel0 = data0.gas.velocities
    vel_x0 = vel0.value[:,0] - vcm[0]
    vel_y0 = vel0.value[:,1] - vcm[1]
    vel_z0 = vel0.value[:,2] - vcm[2]
    m0 = data0.gas.masses.value
    pot0 = data0.gas.potentials.value
    u0 = data0.gas.internal_energies.value
    
    v20 = np.power(vel_x0,2.0) + np.power(vel_y0,2.0) + np.power(vel_z0,2.0)
    ke0 = 0.5*m0*v20
    ie0 = u0*m0
    gpe0 = pot0*m0 * 0.5 #correct for extra factor of two
    
    indexout = np.where(np.isin(part_id0,part_id,invert=True))
    #datapartsort = np.argsort(part_id)
    #dataindexsort = np.argsort(part_id)
    #data0outpartsort = np.argsort(part_id0[indexout])
    
    KElost = np.sum(ke0[indexout],dtype='float64')
    IElost = np.sum(ie0[indexout],dtype='float64')
    GPElost = np.sum(gpe0[indexout],dtype='float64')
    return KElost, IElost, GPElost
    
def find_solvus_mass(data,cm,vcm,index=None):
    if index is None:
        index = np.where(data.gas.particle_ids.value>=0)
    m = data.gas.masses.value[index]
    pot = data.gas.potentials.value[index]
    u = data.gas.internal_energies.value[index]
    P = data.gas.pressures.value[index]
    rho = data.gas.densities.value[index]
    mat_id = data.gas.material_ids.value[index]
    tmp_s = np.zeros(np.size(u))
    tmp_T = np.zeros(np.size(u))
    for i in range(np.size(u)):
        tmp_s[i] = sesame.s_u_rho(u[i],rho[i],mat_id[i])
        tmp_T[i] = sesame.T_rho_s(rho[i],tmp_s[i],mat_id[i])
    
    
    
    

# ----------------------------- # 
dir = "./"
listfilename = 'all_runs_peloton.txt'
listfilename = dir+listfilename
# list of directories and file basename
runlist = np.loadtxt(listfilename,skiprows=1,delimiter='|',dtype={'names':('basename','numsnap','label','write'),'formats':('U256',int,'U256',int)})
Lztotlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Lzbndlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Lzplanetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Lzdisklist = np.empty([np.size(runlist)],dtype=np.ndarray)
Lw0bndlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Lw0totlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Lw0planetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Lw0disklist = np.empty([np.size(runlist)],dtype=np.ndarray)
Nboundlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Nplanetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Ndisklist = np.empty([np.size(runlist)],dtype=np.ndarray)
Nmantlelist = np.empty([np.size(runlist)],dtype=np.ndarray)
Ncorelist = np.empty([np.size(runlist)],dtype=np.ndarray)
Mboundlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Mplanetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Mdisklist = np.empty([np.size(runlist)],dtype=np.ndarray)
Mmantlelist = np.empty([np.size(runlist)],dtype=np.ndarray)
Mcorelist = np.empty([np.size(runlist)],dtype=np.ndarray)
KEtotlist = np.empty([np.size(runlist)],dtype=np.ndarray)
IEtotlist = np.empty([np.size(runlist)],dtype=np.ndarray)
GPEtotlist = np.empty([np.size(runlist)],dtype=np.ndarray)
KEbndlist= np.empty([np.size(runlist)],dtype=np.ndarray)
IEbndlist = np.empty([np.size(runlist)],dtype=np.ndarray)
GPEbndlist = np.empty([np.size(runlist)],dtype=np.ndarray)
KEplanetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
IEplanetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
GPEplanetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
KEdisklist = np.empty([np.size(runlist)],dtype=np.ndarray)
IEdisklist = np.empty([np.size(runlist)],dtype=np.ndarray)
GPEdisklist = np.empty([np.size(runlist)],dtype=np.ndarray)
KEmantlelist = np.empty([np.size(runlist)],dtype=np.ndarray)
IEmantlelist = np.empty([np.size(runlist)],dtype=np.ndarray)
GPEmantlelist = np.empty([np.size(runlist)],dtype=np.ndarray)
KEcorelist = np.empty([np.size(runlist)],dtype=np.ndarray)
IEcorelist = np.empty([np.size(runlist)],dtype=np.ndarray)
GPEcorelist = np.empty([np.size(runlist)],dtype=np.ndarray)
GPEminlist = np.empty([np.size(runlist)])
timelist = np.empty([np.size(runlist)],dtype=np.ndarray)
PGPEtotlist = np.empty([np.size(runlist)],dtype=np.ndarray)
PGPEbndlist = np.empty([np.size(runlist)],dtype=np.ndarray)
PGPEdisklist = np.empty([np.size(runlist)],dtype=np.ndarray)
PGPEplanetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
PGPEmantlelist = np.empty([np.size(runlist)],dtype=np.ndarray)
PGPEcorelist = np.empty([np.size(runlist)],dtype=np.ndarray)
Etotlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Ebndlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Edisklist = np.empty([np.size(runlist)],dtype=np.ndarray)
Eplanetlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Emantlelist = np.empty([np.size(runlist)],dtype=np.ndarray)
Ecorelist = np.empty([np.size(runlist)],dtype=np.ndarray)
Etot0list = np.empty([np.size(runlist)])
Nh_maxtotlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Nh_maxboundlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Ntot0list = np.empty([np.size(runlist)])
Ntotlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Nlostlist = np.empty([np.size(runlist)],dtype=np.ndarray)
KElostlist = np.empty([np.size(runlist)],dtype=np.ndarray)
IElostlist = np.empty([np.size(runlist)],dtype=np.ndarray)
GPElostlist = np.empty([np.size(runlist)],dtype=np.ndarray)
PGPElostlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Elostlist = np.empty([np.size(runlist)],dtype=np.ndarray)
Etot_adjustedlist = np.empty([np.size(runlist)],dtype=np.ndarray)
    
for i in range(np.size(runlist)):
    basename = runlist[i]['basename']
    Nsnap = runlist[i]['numsnap']
    # If we are not set to write a data text file, we check to see if one already exists
    # If a data.txt file exists and gives no errors, we read from it instead of computing everything from hdf5 files
    txtfile = None
    fromhdf5 = True
    if runlist[i]['write']!=1:
        if os.path.isfile(basename+'_data.txt'):
            txtfile = basename+'_data.txt'
    if txtfile is not None:
        try:
            Lztot,Lzbnd,Lzplanet,Lzdisk,Lw0tot,Lw0bnd,Lw0planet,Lw0disk,Ntot,Nbound,Nplanet,Ndisk,Nmantle,Ncore,Mbound,Mplanet,Mdisk,Mmantle,Mcore,KEtot,KEbnd,KEplanet,KEdisk,KEmantle,KEcore,IEtot,IEbnd,IEplanet,IEdisk,IEmantle,IEcore,GPEtot,GPEbnd,GPEplanet,GPEdisk,GPEmantle,GPEcore,PGPEtot,PGPEbnd,PGPEplanet,PGPEdisk,PGPEmantle,PGPEcore,Etot,Ebnd,Eplanet,Edisk,Emantle,Ecore,Nh_maxtot,Nh_maxbound,Nlost,KElost,IElost,GPElost,PGPElost,Elost,Etot_adjusted,Etot0arr,Ntot0arr,time,GPEminarr = np.loadtxt(txtfile,skiprows=1,unpack=True,delimiter='|')
            Etot0 = Etot0arr[0]
            Ntot0 = Ntot0arr[0]
            GPEmin = GPEminarr[0]
            print('Loaded file: ',txtfile)
        except: print('Error opening text file (possibly incomplete), reading from hdf5 instead')
        else:
            if np.size(time)==Nsnap: fromhdf5 = False # Don't need to read from hdf5 files if data text file exists and has all the data
            if os.path.getmtime(basename+'_0000.hdf5')>os.path.getmtime(txtfile):
                fromhdf5 = True # if previous data file is older than the snapshots
        
    if fromhdf5:
        snapshot0 = basename+'_%04d.hdf5' % 0
        data = sw.load(snapshot0)
        data.gas.coordinates.convert_to_mks()
        data.gas.pressures.convert_to_mks()
        data.gas.internal_energies.convert_to_mks()
        data.gas.masses.convert_to_mks()
        data.gas.velocities.convert_to_mks()
        data.gas.smoothing_lengths.convert_to_mks()
        data.gas.densities.convert_to_mks()
        data.gas.potentials.convert_to_mks()
        Ntot = data.gas.metadata.n_gas
        
        cm,vcm = find_initial_com(data)
        print('Initial CoM/vCoM: ',cm,vcm)
    
        Lztot0 = find_Lz(data, cm, vcm)
        print('Initial Lz_tot: ',Lztot0)
        w0 = find_w(data, cm, vcm)
        print('Initial w orientation: ',w0)
        Lw0tot0 = find_Lw(data, cm, vcm, w0)
        print('Initial L_w0: ',Lw0tot0)
        
        Lztot = np.empty([Nsnap])
        Lzbnd = np.empty([Nsnap])
        Lzplanet = np.empty([Nsnap])
        Lzdisk = np.empty([Nsnap])
        Lw0bnd = np.empty([Nsnap])
        Lw0tot = np.empty([Nsnap])
        Lw0planet = np.empty([Nsnap])
        Lw0disk = np.empty([Nsnap])
        Nbound = np.empty([Nsnap])
        Nplanet = np.empty([Nsnap])
        Ndisk = np.empty([Nsnap])
        Nmantle = np.empty([Nsnap])
        Ncore = np.empty([Nsnap])
        Mbound = np.empty([Nsnap])
        Mplanet = np.empty([Nsnap])
        Mdisk = np.empty([Nsnap])
        Mmantle = np.empty([Nsnap])
        Mcore = np.empty([Nsnap])
        KEtot = np.empty([Nsnap])
        IEtot = np.empty([Nsnap])
        GPEtot = np.empty([Nsnap])
        KEbnd= np.empty([Nsnap])
        IEbnd = np.empty([Nsnap])
        GPEbnd = np.empty([Nsnap])
        KEplanet= np.empty([Nsnap])
        IEplanet = np.empty([Nsnap])
        GPEplanet = np.empty([Nsnap])
        KEdisk= np.empty([Nsnap])
        IEdisk = np.empty([Nsnap])
        GPEdisk = np.empty([Nsnap])
        KEmantle= np.empty([Nsnap])
        IEmantle = np.empty([Nsnap])
        GPEmantle = np.empty([Nsnap])
        KEcore= np.empty([Nsnap])
        IEcore = np.empty([Nsnap])
        GPEcore = np.empty([Nsnap])
        GPEmin = 0
        time = np.empty([Nsnap])
        Nh_maxtot = np.empty([Nsnap])
        Nh_maxbound = np.empty([Nsnap])
        Ntot = np.empty([Nsnap])
        Nlost = np.empty([Nsnap])
        KElost = np.empty([Nsnap])
        IElost = np.empty([Nsnap])
        GPElost = np.empty([Nsnap])
        
        Ntot0 = data.gas.metadata.n_gas
        Ntot0list[i]=Ntot0
        
        for snapshot_id in range(0,Nsnap):
            if os.path.isfile(basename+'_%04d.hdf5'%snapshot_id):
                snapshot = basename+'_%04d.hdf5' % snapshot_id
            else:
                snapshot = basename+'_%d.hdf5' % snapshot_id
            
            try:
                data = sw.load(snapshot)
                data.gas.coordinates.convert_to_mks()
            except:
                print('Error loading gas data in file: ',snapshot)
                print('Last stable snapshot will be used')
                stable = False
                j=1
                while not stable:
                    if os.path.isfile(basename+'_%04d.hdf5'%(snapshot_id-j)):
                        snapshot = basename+'_%04d.hdf5' % (snapshot_id-j)
                    else:
                        snapshot = basename+'_%d.hdf5' % (snapshot_id-j)
                    try:
                        data = sw.load(snapshot)
                        data.gas.coordinates.convert_to_mks()
                    except:
                        j+=1
                    else:
                        stable = True
                        print('Last stable snapshot: ',snapshot)
            
            data.gas.pressures.convert_to_mks()
            data.gas.internal_energies.convert_to_mks()
            data.gas.masses.convert_to_mks()
            data.gas.velocities.convert_to_mks()
            data.gas.smoothing_lengths.convert_to_mks()
            data.gas.densities.convert_to_mks()
            data.gas.potentials.convert_to_mks()    
            snaptime = data.metadata.time.in_mks()
            
            print('SNAPSHOT Number: ',snapshot_id)
            print('Simulation time: ',snaptime/3600,' hours')
            
            
            Lztot[snapshot_id] = find_Lz(data,cm,vcm)
            wnew = find_w(data, cm, vcm)
            print('New w orientation: ',wnew)
            Lw0tot[snapshot_id] = find_Lw(data, cm, vcm, wnew)
            time[snapshot_id] = snaptime
            
            print('Total Lz: ',Lztot[snapshot_id])
            print('Total Lw: ',Lw0tot[snapshot_id])
            
            KEtot[snapshot_id], IEtot[snapshot_id], GPEtot[snapshot_id], newGPEmin = find_energy(data, cm, vcm)
            GPEmin = np.amin([GPEmin,newGPEmin])
            print('GPEmin: ',GPEmin)
            
            indbnd, Mbound[snapshot_id], pos, vel = bound_mass(data, cm, vcm)
            Nbound[snapshot_id] = np.size(indbnd)
            #print('indbnd: ',indbnd)
            #databnd = apply_index(data, indbnd) #ERROR - fixing this
            bcm,bvcm = find_initial_com(data,index=indbnd)
            print('Bound mass CoM/vCoM: ',bcm,bvcm)
            wbnd = find_w(data, bcm, bvcm, index=indbnd)
            print('Bound w orientation: ',wbnd)
            
            indexdisk, indexplanet, indexmantle, indexcore, Mdisk[snapshot_id], Mplanet[snapshot_id], Mmantle[snapshot_id], Mcore[snapshot_id] = find_disk(data, cm, vcm, pos, vel,index=indbnd)
            #datadisk = apply_index(databnd, inddisk)
            try:
                inddisk = np.asarray(indbnd)[0][indexdisk]
                Lzdisk[snapshot_id] = find_Lz(data,cm,vcm,index=inddisk)
                Lw0disk[snapshot_id] = find_Lw(data,bcm,bvcm,wbnd,index=inddisk)
                KEdisk[snapshot_id], IEdisk[snapshot_id], GPEdisk[snapshot_id], temp = find_energy(data, cm, vcm,index=inddisk)
            except:
                inddisk = np.array([])
                Lzdisk[snapshot_id]=0
                Lw0disk[snapshot_id]=0
                KEdisk[snapshot_id], IEdisk[snapshot_id], GPEdisk[snapshot_id] = 0,0,GPEmin
            try:
                indplanet = np.asarray(indbnd)[0][indexplanet]
                Lzplanet[snapshot_id] = find_Lz(data,cm,vcm,index=indplanet)
                Lw0planet[snapshot_id] = find_Lw(data,bcm,bvcm,wbnd,index=indplanet)
                KEplanet[snapshot_id], IEplanet[snapshot_id], GPEplanet[snapshot_id], temp = find_energy(data, cm, vcm,index=indplanet)
            except:
                indplanet = np.array([])
                Lzplanet[snapshot_id] = 0
                Lw0planet[snapshot_id] = 0
                KEplanet[snapshot_id], IEplanet[snapshot_id], GPEplanet[snapshot_id] = 0,0,GPEmin
            try:
                indmantle = np.asarray(indbnd)[0][indexmantle]
                KEmantle[snapshot_id], IEmantle[snapshot_id], GPEmantle[snapshot_id], temp = find_energy(data, cm, vcm,index=indmantle)
            except:
                indmantle = np.array([])
                KEmantle[snapshot_id], IEmantle[snapshot_id], GPEmantle[snapshot_id] = 0,0,GPEmin
            try:
                indcore = np.asarray(indbnd)[0][indexcore]
                KEcore[snapshot_id], IEcore[snapshot_id], GPEcore[snapshot_id], temp = find_energy(data, cm, vcm,index=indcore)
            except:
                indcore = np.array([])
                KEcore[snapshot_id], IEcore[snapshot_id], GPEcore[snapshot_id] = 0,0,GPEmin
            Ndisk[snapshot_id] = np.size(inddisk)
            #print('indplanet: ',indplanet)
            #print('inddisk: ',inddisk)
            #dataplanet = apply_index(databnd, indplanet)
            Nplanet[snapshot_id] = np.size(indplanet)
            #print('indmantle: ',indmantle)
            #datamantle = apply_index(databnd, indmantle)
            Nmantle[snapshot_id] = np.size(indmantle)
            #print('indcore: ',indcore)
            #datacore = apply_index(databnd, indcore)
            Ncore[snapshot_id] = np.size(indcore)
            
            Lzbnd[snapshot_id] = find_Lz(data,cm,vcm,index=indbnd)
            Lw0bnd[snapshot_id] = find_Lw(data,bcm,bvcm,wbnd,index=indbnd)
            # Lzdisk[snapshot_id] = find_Lz(data,cm,vcm,index=inddisk)
            # Lw0disk[snapshot_id] = find_Lw(data,cm,vcm,w0,index=inddisk)
            # Lzplanet[snapshot_id] = find_Lz(data,cm,vcm,index=indplanet)
            # Lw0planet[snapshot_id] = find_Lw(data,cm,vcm,w0,index=indplanet)
            
            print('Bound Lz: ',Lzbnd[snapshot_id])
            print('Bound Lw0: ',Lw0bnd[snapshot_id])
            print('Disk Lz: ',Lzdisk[snapshot_id])
            print('Disk Lw0: ',Lw0disk[snapshot_id])
            print('Planet Lz: ',Lzplanet[snapshot_id])
            print('Planet Lw0: ',Lw0planet[snapshot_id])
            
            KEbnd[snapshot_id], IEbnd[snapshot_id], GPEbnd[snapshot_id], temp = find_energy(data, cm, vcm,index=indbnd)
            #KEdisk[snapshot_id], IEdisk[snapshot_id], GPEdisk[snapshot_id], temp = find_energy(data, cm, vcm,index=inddisk)
            #KEplanet[snapshot_id], IEplanet[snapshot_id], GPEplanet[snapshot_id], temp = find_energy(data, cm, vcm,index=indplanet)
            #KEmantle[snapshot_id], IEmantle[snapshot_id], GPEmantle[snapshot_id], temp = find_energy(data, cm, vcm,index=indmantle)
            #KEcore[snapshot_id], IEcore[snapshot_id], GPEcore[snapshot_id], temp = find_energy(data, cm, vcm,index=indcore)
            
            Ntot[snapshot_id] = data.gas.metadata.n_gas
            Nlost[snapshot_id] = Ntot0 - data.gas.metadata.n_gas
            Nh_maxtot[snapshot_id] = find_h_max_parts(data)
            Nh_maxbound[snapshot_id] = find_h_max_parts(data,index=indbnd)
            print('N total particles at density floor: ',Nh_maxtot[snapshot_id])
            print('N bound particles at density floor: ',Nh_maxbound[snapshot_id])
            print('N lost particles: ',Nlost[snapshot_id])
            if snapshot_id > 0:
                lostke, lostie, lostgpe = find_lost_energy(data, cm, vcm, snapshot_id, basename)
                KElost[snapshot_id] = lostke + KElost[snapshot_id-1]
                IElost[snapshot_id] = lostie + IElost[snapshot_id-1]
                GPElost[snapshot_id] = lostgpe + GPElost[snapshot_id-1]
            else:
                KElost[snapshot_id] = 0
                IElost[snapshot_id] = 0
                GPElost[snapshot_id] = 0
            
            
        PGPEtot = GPEtot - GPEmin*Ntot
        PGPEbnd = GPEbnd - GPEmin*Nbound
        PGPEdisk = GPEdisk - GPEmin*Ndisk
        PGPEplanet = GPEplanet - GPEmin*Nplanet
        PGPEmantle = GPEmantle - GPEmin*Nmantle
        PGPEcore = GPEcore - GPEmin*Ncore
        PGPElost = GPElost - GPEmin*Nlost
        Etot = PGPEtot + KEtot + IEtot
        Ebnd = PGPEbnd + KEbnd + IEbnd
        Edisk = PGPEdisk + KEdisk + IEdisk
        Eplanet = PGPEplanet + KEplanet + IEplanet
        Emantle = PGPEmantle + KEmantle + IEmantle
        Ecore = PGPEcore + KEcore + IEcore
        Elost = KElost + IElost + PGPElost
        Etot_adjusted = Etot+Elost
        Etot0 = Etot[0]
        
    Lztotlist[i] = Lztot
    Lzbndlist[i] = Lzbnd
    Lzplanetlist[i] = Lzplanet
    Lzdisklist[i] = Lzdisk
    Lw0bndlist[i] = Lw0bnd
    Lw0totlist[i] = Lw0tot
    Lw0planetlist[i] = Lw0planet
    Lw0disklist[i] = Lw0disk
    Ntotlist[i] = Ntot
    Ntot0list[i] = Ntot0
    Nboundlist[i] = Nbound
    Nplanetlist[i] = Nplanet
    Ndisklist[i] = Ndisk
    Nmantlelist[i] = Nmantle
    Ncorelist[i] = Ncore
    Mboundlist[i] = Mbound
    Mplanetlist[i] = Mplanet
    Mdisklist[i] = Mdisk
    Mmantlelist[i] = Mmantle
    Mcorelist[i] = Mcore
    KEtotlist[i] = KEtot
    IEtotlist[i] = IEtot
    GPEtotlist[i] = GPEtot
    KEbndlist[i] = KEbnd
    IEbndlist[i] = IEbnd
    GPEbndlist[i] = GPEbnd
    KEplanetlist[i] = KEplanet
    IEplanetlist[i] = IEplanet
    GPEplanetlist[i] = GPEplanet
    KEdisklist[i] = KEdisk
    IEdisklist[i] = IEdisk
    GPEdisklist[i] = GPEdisk
    KEmantlelist[i] = KEmantle
    IEmantlelist[i] = IEmantle
    GPEmantlelist[i] = GPEmantle
    KEcorelist[i] = KEcore
    IEcorelist[i] = IEcore
    GPEcorelist[i] = GPEcore
    GPEminlist[i] = GPEmin
    timelist[i] = time
    PGPEtotlist[i] = PGPEtot
    PGPEbndlist[i] = PGPEbnd
    PGPEdisklist[i] = PGPEdisk
    PGPEplanetlist[i] = PGPEplanet
    PGPEmantlelist[i] = PGPEmantle
    PGPEcorelist[i] = PGPEcore
    Etotlist[i] = Etot
    Ebndlist[i] = Ebnd
    Edisklist[i] = Edisk
    Eplanetlist[i] = Eplanet
    Emantlelist[i] = Emantle
    Ecorelist[i] = Ecore
    Etot0list[i] = Etot0
    Nh_maxtotlist[i] = Nh_maxtot
    Nh_maxboundlist[i] = Nh_maxbound
    Nlostlist[i] = Nlost
    KElostlist[i] = KElost
    IElostlist[i] = IElost
    GPElostlist[i] = GPElost
    PGPElostlist[i] = PGPElost
    Elostlist[i] = Elost
    Etot_adjustedlist[i] = Etot_adjusted       
    
    if (runlist[i]['write']==1) or fromhdf5:
        with open(basename+'_data.txt','w') as writefile:
            writefile.write('#Lztot|Lzbnd|Lzplanet|Lzdisk|Lw0tot|Lw0bnd|Lw0planet|Lw0disk|Ntot|Nbnd|Nplanet|Ndisk|Nmantle|Ncore|Mbnd|Mplanet|Mplanet|Mdisk|Mmantle|Mcore|KEtot|KEbnd|KEplanet|KEdisk|KEmantle|KEcore|IEtot|IEbnd|IEplanet|IEdisk|IEmantle|IEcore|GPEtot|GPEbnd|GPEplanet|GPEdisk|GPEmantle|GPEcore|PGPEtot|PGPEbnd|PGPEplanet|PGPEdisk|PGPEmantle|PGPEcore|Etot|Ebnd|Eplanet|Edisk|Emantle|Ecore|Nh_maxtot|Nh_maxbnd|Nlost|KElost|IElost|GPElost|PGPElost|Elost|Etot_adjusted|Etot0|Ntot0|time|GPEmin \n')
            for snap_id in range(0,Nsnap):
                #writefile.write(Lztot[snap_id]+'|'+Lzbnd[snap_id]+'|'+Lzplanet[snap_id]+'|'+Lzdisk[snap_id]+'|'+Lw0tot[snap_id]+'|'+Lw0bnd[snap_id]+'|'+Lw0planet[snap_id]+'|'+Lw0disk[snap_id]+'|'+Ntot[snap_id]+'|'+Nbound[snap_id]+'|'+Nplanet[snap_id]+'|'+Ndisk[snap_id]+'|'+Nmantle[snap_id]+'|'+Ncore[snap_id]+'|'+Mbound[snap_id]+'|'+Mplanet[snap_id]+'|'+Mdisk[snap_id]+'|'+Mmantle[snap_id]+'|'+Mcore[snap_id]+'|'+KEtot[snap_id]+'|'+KEbnd[snap_id]+'|'+KEplanet[snap_id]+'|'+KEdisk[snap_id]+'|'+KEmantle[snap_id]+'|'+KEcore[snap_id]+'|'+IEtot[snap_id]+'|'+IEbnd[snap_id]+'|'+IEplanet[snap_id]+'|'+IEdisk[snap_id]+'|'+IEmantle[snap_id]+'|'+IEcore[snap_id]+'|'+GPEtot[snap_id]+'|'+GPEbnd[snap_id]+'|'+GPEplanet[snap_id]+'|'+GPEdisk[snap_id]+'|'+GPEmantle[snap_id]+'|'+GPEcore[snap_id]+'|'+PGPEtot[snap_id]+'|'+PGPEbnd[snap_id]+'|'+PGPEplanet[snap_id]+'|'+PGPEdisk[snap_id]+'|'+PGPEmantle[snap_id]+'|'+PGPEcore[snap_id]+'|'+Etot[snap_id]+'|'+Ebnd[snap_id]+'|'+Eplanet[snap_id]+'|'+Edisk[snap_id]+'|'+Emantle[snap_id]+'|'+Ecore[snap_id]+'|'+Nh_maxtot[snap_id]+'|'+Nh_maxbound[snap_id]+'|'+Nlost[snap_id]+'|'+KElost[snap_id]+'|'+IElost[snap_id]+'|'+GPElost[snap_id]+'|'+PGPElost[snap_id]+'|'+Elost[snap_id]+'|'+Etot_adjusted[snap_id]+'|'+Etot0+'|'+Ntot0+'\n')
                writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(Lztot[snap_id],Lzbnd[snap_id],Lzplanet[snap_id],Lzdisk[snap_id],Lw0tot[snap_id],Lw0bnd[snap_id],Lw0planet[snap_id],Lw0disk[snap_id],Ntot[snap_id],Nbound[snap_id],Nplanet[snap_id],Ndisk[snap_id],Nmantle[snap_id],Ncore[snap_id],Mbound[snap_id],Mplanet[snap_id],Mdisk[snap_id],Mmantle[snap_id],Mcore[snap_id],KEtot[snap_id],KEbnd[snap_id],KEplanet[snap_id],KEdisk[snap_id],KEmantle[snap_id],KEcore[snap_id],IEtot[snap_id],IEbnd[snap_id],IEplanet[snap_id],IEdisk[snap_id],IEmantle[snap_id],IEcore[snap_id],GPEtot[snap_id],GPEbnd[snap_id],GPEplanet[snap_id],GPEdisk[snap_id],GPEmantle[snap_id],GPEcore[snap_id],PGPEtot[snap_id],PGPEbnd[snap_id],PGPEplanet[snap_id],PGPEdisk[snap_id],PGPEmantle[snap_id],PGPEcore[snap_id],Etot[snap_id],Ebnd[snap_id],Eplanet[snap_id],Edisk[snap_id],Emantle[snap_id],Ecore[snap_id],Nh_maxtot[snap_id],Nh_maxbound[snap_id],Nlost[snap_id],KElost[snap_id],IElost[snap_id],GPElost[snap_id],PGPElost[snap_id],Elost[snap_id],Etot_adjusted[snap_id],Etot0,Ntot0,time[snap_id],GPEmin)+'\n')
        print('Saved file ',basename+'_data.txt')
        
    fir2, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.plot(time/3600,(Lztot-Lztot[0])/Lztot[0],label='Lz, initial Lz=%04f LEM'%Lztot[0])
    ax.plot(time/3600,(Lw0tot-Lw0tot[0])/Lw0tot[0],label='Lw0, initial Lw0=%04f LEM'%Lw0tot[0])
    ax.set_xlabel('time (hours)')
    ax.set_ylabel('Fractional change in total angular momentum (LEM)')
    #ax.axis('equal')
    #ax.set_ylim(2.5,3.5)
    ax.legend()
    plt.tight_layout()
    imname = basename+'_L_tot.png'
    plt.savefig(imname, dpi=100)
    plt.close()

    fir2, ax = plt.subplots(1, 1, figsize=(10,6))
    ax.plot(time/3600,Lzbnd,label='Lz of bound mass')
    ax.plot(time/3600,Lw0bnd,label='Lw0 of bound mass')
    ax.plot(time/3600,Lzplanet,label='Lz of planet mass')
    ax.plot(time/3600,Lw0planet,label='Lw0 of planet mass')
    ax.plot(time/3600,Lzdisk,label='Lz of disk mass')
    ax.plot(time/3600,Lw0disk,label='Lw0 of disk mass')
    ax.set_xlabel('time (hours)')
    ax.set_ylabel('Total bound angular momentum (LEM)')
    ax.legend(fontsize=8)
    plt.tight_layout()
    imname = basename+'_L_bound.png'
    plt.savefig(imname, dpi=100)
    plt.close()

    fir2, ax = plt.subplots(4, 1, figsize=(10,20))
    ax[0].plot(time/3600,Etot/Etot0,label='Total energy (PGPE+KE+IE)')
    ax[0].plot(time/3600,PGPEtot/Etot0,label='Participating GPE')
    ax[0].plot(time/3600,KEtot/Etot0,label='KE')
    ax[0].plot(time/3600,IEtot/Etot0,label='IE')
    ax[0].set_xlabel('time (hours)')
    ax[0].set_ylabel('Fraction of total initial energy')
    ax[0].legend(fontsize=8)
    ax[0].set_title('Total system energy')

    ax[1].plot(time/3600,Ebnd/Etot0,label='Bound energy (PGPEbnd+KEbnd+IEbnd)')
    ax[1].plot(time/3600,PGPEbnd/Etot0,label='Participating GPEbnd')
    ax[1].plot(time/3600,KEbnd/Etot0,label='KEbnd')
    ax[1].plot(time/3600,IEbnd/Etot0,label='IEbnd')
    ax[1].plot(time/3600,Eplanet/Etot0,label='Planet energy (PGPEplanet+KEplanet+IEplanet)')
    ax[1].plot(time/3600,PGPEplanet/Etot0,label='Participating GPEplanet')
    ax[1].plot(time/3600,KEplanet/Etot0,label='KEplanet')
    ax[1].plot(time/3600,IEplanet/Etot0,label='IEplanet')
    ax[1].plot(time/3600,Edisk/Etot0,label='Disk energy (PGPEdisk+KEdisk+IEdisk)')
    ax[1].plot(time/3600,PGPEdisk/Etot0,label='Participating GPEdisk')
    ax[1].plot(time/3600,KEdisk/Etot0,label='KEdisk')
    ax[1].plot(time/3600,IEdisk/Etot0,label='IEdisk')
    ax[1].set_xlabel('time (hours)')
    ax[1].set_ylabel('Fraction of total initial energy')
    ax[1].legend(fontsize=8)
    ax[1].set_title('Bound mass energy')

    ax[2].plot(time/3600,Eplanet/Etot0,label='Planet energy (PGPEplanet+KEplanet+IEplanet)')
    ax[2].plot(time/3600,PGPEplanet/Etot0,label='Participating GPEplanet')
    ax[2].plot(time/3600,KEplanet/Etot0,label='KEplanet')
    ax[2].plot(time/3600,IEplanet/Etot0,label='IEplanet')
    ax[2].plot(time/3600,Emantle/Etot0,label='Mantle energy (PGPEmantle+KEmantle+IEmantle)')
    ax[2].plot(time/3600,PGPEmantle/Etot0,label='Participating GPEmantle')
    ax[2].plot(time/3600,KEmantle/Etot0,label='KEmantle')
    ax[2].plot(time/3600,IEmantle/Etot0,label='IEmantle')
    ax[2].plot(time/3600,Ecore/Etot0,label='Core energy (PGPEcore+KEcore+IEcore)')
    ax[2].plot(time/3600,PGPEcore/Etot0,label='Participating GPEcore')
    ax[2].plot(time/3600,KEcore/Etot0,label='KEcore')
    ax[2].plot(time/3600,IEcore/Etot0,label='IEcore')
    ax[2].set_xlabel('time (hours)')
    ax[2].set_ylabel('Fraction of total initial energy')
    ax[2].legend(fontsize=8)
    ax[2].set_title('Planet energy')
    
    Mplanet = np.where(Mplanet==0,np.nextafter(np.float32(0), np.float32(1)),Mplanet)
    ax[3].plot(time/3600,IEmantle/Etot0/(Mmantle/Mplanet),label='scaled IEmantle/Mmantle')
    ax[3].plot(time/3600,IEcore/Etot0/(Mcore/Mplanet),label='scaled IEcore/Mcore')
    ax[3].set_xlabel('time (hours)')
    ax[3].set_ylabel('Fraction of total initial energy per mass fraction')
    ax[3].legend(fontsize=8)
    ax[3].set_title('Scaled internal energy per mass fraction')

    plt.tight_layout()
    imname = basename+'_energies.png'
    plt.savefig(imname, dpi=100)
    #print('Saved image: ',imname)
    plt.close()
    
    fig1,ax = plt.subplots(3, 1, figsize=(10,18))
    ax[0].plot(time/3600,Etot_adjusted/Etot0,label=runlist[i]['label']+' (corrected for lost particles)')
        #ax[0].plot(timelist[i]/3600,Etotlist[i]/Etot0list[i],label=runlist[i]['label']+' (unadjusted)',ls='--')
    ax[0].set_xlabel('time (hours)')
    ax[0].set_ylabel('Fraction of total initial energy')
    ax[0].legend(fontsize=8)
    ax[0].set_title('Total energy conservation')

    ax[1].plot(time/3600,(Nh_maxtot)/Ntot0,label=runlist[i]['label']+' total h_max particles')
    ax[1].plot(time/3600,(Nlost)/Ntot0,label=runlist[i]['label']+' total lost particles',ls='--')
    ax[1].set_xlabel('time (hours)')
    ax[1].set_ylabel('Fraction of particles')
    ax[1].legend(fontsize=8)
    ax[1].set_title('Particles lost or at density floor')

    ax[2].plot(time/3600,Nlost/Ntot0,label=runlist[i]['label']+' lost particles',ls='--')
    ax[2].plot(time/3600,(Nh_maxtot-Nh_maxbound)/Ntot0,label=runlist[i]['label']+' unbound h_max particles',ls='-.')
    ax[2].plot(time/3600,(Nh_maxbound)/Ntot0,label=runlist[i]['label']+' bound h_max particles',ls=':')
    ax[2].set_xlabel('time (hours)')
    ax[2].set_ylabel('Fraction of particles')
    ax[2].legend(fontsize=8)
    ax[2].set_title('Particles lost or at density floor')

    plt.tight_layout()
    imname = basename+'_energy_conservation.png'
    plt.savefig(imname, dpi=100)
    print('Saved image: ',imname)
    plt.close()

    
fig1,ax = plt.subplots(3, 1, figsize=(10,18))
for i in range(np.size(runlist)):
    ax[0].plot(timelist[i]/3600,Etot_adjustedlist[i]/Etot0list[i],label=runlist[i]['label']+' (corrected for lost particles)')
    #ax[0].plot(timelist[i]/3600,Etotlist[i]/Etot0list[i],label=runlist[i]['label']+' (unadjusted)',ls='--')
ax[0].set_xlabel('time (hours)')
ax[0].set_ylabel('Fraction of total initial energy')
ax[0].legend(fontsize=8)
ax[0].set_title('Total energy conservation')

for i in range(np.size(runlist)):
    ax[1].plot(timelist[i]/3600,(Nh_maxtotlist[i])/Ntot0list[i],label=runlist[i]['label']+' total h_max particles')
    ax[1].plot(timelist[i]/3600,(Nlostlist[i])/Ntot0list[i],label=runlist[i]['label']+' total lost particles',ls='--')
ax[1].set_xlabel('time (hours)')
ax[1].set_ylabel('Fraction of particles')
ax[1].legend(fontsize=8)
ax[1].set_title('Particles lost or at density floor')

for i in range(np.size(runlist)):
    ax[2].plot(timelist[i]/3600,Nlostlist[i]/Ntot0list[i],label=runlist[i]['label']+' lost particles',ls='--')
    ax[2].plot(timelist[i]/3600,(Nh_maxtotlist[i]-Nh_maxboundlist[i])/Ntot0list[i],label=runlist[i]['label']+' unbound h_max particles',ls='-.')
    ax[2].plot(timelist[i]/3600,(Nh_maxboundlist[i])/Ntot0list[i],label=runlist[i]['label']+' bound h_max particles',ls=':')
ax[2].set_xlabel('time (hours)')
ax[2].set_ylabel('Fraction of particles')
ax[2].legend(fontsize=8)
ax[2].set_title('Particles lost or at density floor')

imname = dir+'energy_conservation.png'
plt.tight_layout()
plt.savefig(imname, dpi=100)
print('Saved image: ',imname)
plt.close()
    
