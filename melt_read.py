# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 13:05:30 2024

@author: adriana
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
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

eospath = 'C:/Users/gerri/'
sys.path.append('C:/Users/gerri/aneos-pyrolite-2022/')
from eostable_dev import *

#eospath = '/home/apostema/aneos/'
#sys.path.append('/home/apostema/aneos/aneos-pyrolite-2022/')
#from eostable_dev import *

R_earth = gv.R_earth #6.371e6   # m
M_earth = gv.M_earth #5.9724e24  # kg 
G = gv.G #6.67408e-11  # m^3 kg^-1 s^-2
LEM=3.5E34 #AMof Earth-Moon system in mks

mant_mat_id = 403 #USER INPUT
core_mat_id = 402 #USER INPUT

def loadEOS(eos='Iron-ANEOS-SLVTv0.2G1'):
# READ IN NEW ANEOS MODEL and fill the extEOStable class object
# source in eostable.py
#------------------------------------------------------------------

    if eos == 'Iron-ANEOS-SLVTv0.2G1':
        eosdir = eospath + 'aneos-iron-2020/'

        MODELNAME = 'Iron-ANEOS-SLVTv0.2G1'
        # Header information must all be compatible with float format
        MATID = 1.0        # MATID number
        DATE = 191105.     # Date as a single 6-digit number YYMMDD
        VERSION = 0.2      # ANEOS Parameters Version number
        FMN = 26.          # Formula weight in atomic numbers for Fe
        FMW = 55.847       # Formula molecular weight (g/cm3) for Fe
        # The following define the default initial state for material in the 201 table
        R0REF   = 8.06     # g/cm3 *** R0REF is inserted into the density array; using gamma-iron for rho0
        K0REF   = 1.51E12  # dynes/cm2; using gamma-iron for rho0
        T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
        P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below

    elif eos == 'Fe85Si15-ANEOS-SLVTv0.2G1':
        eosdir = eospath + 'aneos-Fe85Si15-2020/'
        # ====>>>>>> YOU NEED TO MAKE SURE THESE VALUES MATCH ANEOS.INPUT  <<<<=====
        MODELNAME = 'Fe85Si15-ANEOS-SLVTv0.2G1'
        # Header information must all be compatible with float format
        MATID = 1.0        # MATID number
        DATE = 191105.     # Date as a single 6-digit number YYMMDD
        VERSION = 0.2      # ANEOS Parameters Version number
        FMN = 24.20        # Formula weight in atomic numbers for Fe85Si15
        FMW = 51.68        # Formula molecular weight (g/cm3) for Fe85Si15
        # The following define the default initial state for material in the 201 table
        R0REF   = 7.51     # g/cm3 *** R0REF is inserted into the density array; using gamma-iron for rho0
        K0REF   = 1.51E12  # dynes/cm2; using gamma-iron for rho0
        T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
        P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below
        #-------------------------------------------------------------

    elif eos == 'Forsterite-ANEOS-SLVTv1.0G1':
        eosdir = eospath + 'aneos-forsterite-2019/'
        MODELNAME = 'Forsterite-ANEOS-SLVTv1.0G1'
        # Header information must all be compatible with float format
        MATID = 1.0        # MATID number
        DATE = 190802.     # Date as a single 6-digit number YYMMDD
        VERSION = 0.1      # ANEOS Parameters Version number
        FMN = 70.          # Formula weight in atomic numbers for Mg2SiO4
        FMW = 140.691      # Formula molecular weight (g/cm3) for Mg2SiO4
        # The following define the default initial state for material in the 201 table
        R0REF   = 3.22     # g/cm3 *** R0REF is inserted into the density array
        K0REF   = 1.1E12   # dynes/cm2
        T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
        P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below
        
    elif eos == 'Pyrolite-ANEOS-SLVTv0.2G1':
        eosdir = eospath + '/aneos-pyrolite-2022//'
        MODELNAME = 'Pyrolite-ANEOS-SLVTv0.2G1'
        # Header information must all be compatible with float format
        MATID = 1.0        # MATID number
        DATE = 210627.     # Date as a single 6-digit number YYMMDD
        VERSION = 0.1      # ANEOS Parameters Version number
        FMN = 153.          # Formula weight in atomic numbers for Mg2SiO4
        FMW = 3234.61      # Formula molecular weight (g/cm3) for Mg2SiO4
        # The following define the default initial state for material in the 201 table
        R0REF   = 3.35     # g/cm3 *** R0REF is inserted into the density array
        K0REF   = 0.95E12  # dynes/cm2
        T0REF   = 298.     # K -- *** T0REF is inserted into the temperature array
        P0REF   = 1.E6     # dynes/cm2 -- this defines the principal Hugoniot calculated below
    
    
    
    NewEOS  = extEOStable() # FIRST make new empty EOS object
    NewEOS.loadextsesame(eosdir+'NEW-SESAME-EXT.TXT') # LOAD THE EXTENDED 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
    NewEOS.loadstdsesame(eosdir+'NEW-SESAME-STD.TXT') # LOAD THE STANDARD 301 SESAME FILE GENERATED BY STSM VERSION OF ANEOS
    NewEOS.MODELNAME = MODELNAME # string set above in user input
    NewEOS.MDQ = np.zeros((NewEOS.NT,NewEOS.ND)) # makes the empty MDQ array
    #print(NewEOS.units) # these are the default units for SESAME rho-T tables
    #'Units: g/cm3, K, GPa, MJ/kg, MJ/kg, MJ/K/kg, cm/s, MJ/K/kg, KPA flag. 2D arrays are (NT,ND).'

    # Add the header info to the table. This could be done during the loading. 
    # if made from this notebook, these values are set in the user-input above.
    # ** MAKE SURE THEY MATCH ANEOS.INPUT **
    NewEOS.MATID   = MATID
    NewEOS.DATE    = DATE
    NewEOS.VERSION = VERSION
    NewEOS.FMN     = FMN
    NewEOS.FMW     = FMW
    NewEOS.R0REF   = R0REF
    NewEOS.K0REF   = K0REF
    NewEOS.T0REF   = T0REF
    NewEOS.P0REF   = P0REF
    #
    # Load the information from ANEOS.INPUT and ANEOS.OUTPUT
    NewEOS.loadaneos(aneosinfname=eosdir+'ANEOS.INPUT',aneosoutfname=eosdir+'ANEOS.OUTPUT')#,silent=True)
    #
    NewEOS.calchugoniot(r0=NewEOS.R0REF,t0=NewEOS.T0REF)#,silent=True)
    #
    # calculate the 1-bar profile; loop over temp
    NewEOS.onebar.T = np.zeros(NewEOS.NT)
    NewEOS.onebar.S = np.zeros(NewEOS.NT)
    NewEOS.onebar.rho = np.zeros(NewEOS.NT)
    it0 = np.where(NewEOS.T >= NewEOS.T0REF)[0]
    id0 = np.arange(NewEOS.ND)#np.where(NewEOS.rho >= 0.8*NewEOS.R0REF)[0]
    for iit in range(0,NewEOS.NT):
        NewEOS.onebar.T[iit] = NewEOS.T[iit]
        NewEOS.onebar.S[iit] = np.interp(1.E-4,NewEOS.P[iit,id0],NewEOS.S[iit,id0])
        NewEOS.onebar.rho[iit] = np.interp(1.E-4,NewEOS.P[iit,id0],NewEOS.rho[id0])

    #print(NewEOS.vc.rl)

    return NewEOS

if core_mat_id == 401: CoreEOS = loadEOS(eos='Iron-ANEOS-SLVTv0.2G1')
if core_mat_id == 402: CoreEOS = loadEOS(eos='Fe85Si15-ANEOS-SLVTv0.2G1')
if mant_mat_id == 400: MantEOS = loadEOS(eos='Forsterite-ANEOS-SLVTv1.0G1')
if mant_mat_id == 403: MantEOS = loadEOS(eos='Pyrolite-ANEOS-SLVTv0.2G1')

def find_phase(data,cm,vcm,pos,vel,rplanet):
    '''
    Finds ANEOS matter phase:
        Solid: if T < triple pt, rho > liquid melt curve extension, if T > triple pt, rho > solid melt curve
        Solid + vapor: T < triple pt, rho between gas and liquid vapor curves (or T below both)
        Melt: T > triple pt, T < critical pt, rho between liquid melt and solid melt curves    
        Liquid: T > triple pt, T < critical pt, rho between liquid vapor and liquid melt curves
        Liquid + vapor: T > triple pt, T < critical pt, rho between gas vapor and liquid vapor curves
        Vapor: T > vapor curve, T < critical pt, rho < gas vapor curve
        Gas: T > critical pt, P < critical pt
        Supercritical: T > critical pt, T > liquid melt curve, P > critical pt, rho < liquid melt curve

    Parameters
    ----------
    data : swift dataset
        DESCRIPTION.

    Returns
    -------
    1: Solid
    2: Solid + vapor
    3: Melt
    4: Liquid
    5: Liquid + vapor
    6: Vapor
    7: Gas
    8: Supercritical fluid

    '''
    mat_id = data.gas.material_ids.value
    u = data.gas.internal_energies.value
    m = data.gas.masses.value
    rho = data.gas.densities.value
    Ntot=np.size(u)
    tmp_s = np.zeros(Ntot)
    tmp_T = np.zeros(Ntot)
    for i in range(np.size(u)):
        tmp_s[i] = sesame.s_u_rho(u[i],rho[i],mat_id[i])
        tmp_T[i] = sesame.T_rho_s(rho[i],tmp_s[i],mat_id[i])
    
    rhocgs = rho/1.e3 #CONVERT TO CGS TO MATCH ANEOS OUTPUT ARRAYS
    P = data.gas.pressures.value/1.e9 #CONVERT TO GPa
    
    boxsize = data.metadata.boxsize.value
    pos_x = pos.value[:,0]-boxsize[0]/2*R_earth #- cm[0]
    pos_y = pos.value[:,1]-boxsize[1]/2*R_earth #- cm[1]
    pos_z = pos.value[:,2]-boxsize[2]/2*R_earth #- cm[2]
    
    vel_x = vel.value[:,0] #- vcm[0]
    vel_y = vel.value[:,1] #- vcm[1]
    vel_z = vel.value[:,2] #- vcm[2]
    
    rrr = np.sqrt((pos_x)**2 + (pos_y)**2 + (pos_z)**2)
    rr = np.sqrt((pos_x)**2 + (pos_y)**2)
    zlim = rplanet*R_earth/10
    
    
    phase = np.zeros(Ntot)
    
    #interpolation functions
    # vapor curves
    mantVcRl = interpolate.interp1d(MantEOS.vc.T,MantEOS.vc.rl)
    mantVcRv = interpolate.interp1d(MantEOS.vc.T,MantEOS.vc.rv)
    coreVcRl = interpolate.interp1d(CoreEOS.vc.T,CoreEOS.vc.rl)
    coreVcRv = interpolate.interp1d(CoreEOS.vc.T,CoreEOS.vc.rv)
    # melt curves
    mantMcRl = interpolate.interp1d(MantEOS.mc.T, MantEOS.mc.rl)
    mantMcRs = interpolate.interp1d(MantEOS.mc.T, MantEOS.mc.rs)
    coreMcRl = interpolate.interp1d(CoreEOS.mc.T, CoreEOS.mc.rl)
    coreMcRs = interpolate.interp1d(CoreEOS.mc.T, CoreEOS.mc.rs)
    
    # BEGIN PHASE DECISION TREE
    for i in range(Ntot):
        # Select EOS to use
        if mat_id[i] == mant_mat_id:
            useEOS = MantEOS
            VcRl = mantVcRl
            VcRv = mantVcRv
            McRl = mantMcRl
            McRs = mantMcRs
        if mat_id[i] == core_mat_id:
            useEOS = CoreEOS
            VcRl = coreVcRl
            VcRv = coreVcRv
            McRl = coreMcRl
            McRs = coreMcRs
            
        # T less than bottom of vapor curve
        if tmp_T[i] <= np.amin(useEOS.vc.T):
            if rhocgs[i] <= np.amax(useEOS.mc.rl): phase[i]=2   # solid + vapor
            else: phase[i]=1                                    # solid
        # T between bottom of vapor curve and triple point
        elif tmp_T[i] <= useEOS.tp.T:
            if rhocgs[i] <= VcRv(tmp_T[i]): phase[i]=6          # vapor
            elif rhocgs[i] <= VcRl(tmp_T[i]): phase[i]=2        # solid + vapor
            else: phase[i]=1                                    # solid
        # T between triple point and critical point
        elif tmp_T[i] <= useEOS.cp.T:
            if rhocgs[i] <= VcRv(tmp_T[i]): phase[i]=6          # vapor
            elif rhocgs[i] <= VcRl(tmp_T[i]): phase[i]=5        # liquid + vapor
            elif rhocgs[i] <= McRl(tmp_T[i]): phase[i]=4        # liquid
            elif rhocgs[i] <= McRs(tmp_T[i]): phase[i]=3        # melt
            else: phase[i]=1                                    # solid
        # T between triple point and top of melt curve
        elif tmp_T[i] <= np.amax(useEOS.mc.T):
            if P[i] <= useEOS.cp.P: phase[i]=7                  # gas
            elif rhocgs[i] <= McRl(tmp_T[i]): phase[i]=8        # supercritical fluid
            elif rhocgs[i] <= McRs(tmp_T[i]): phase[i]=3        # melt
            else: phase[i]=1                                    # solid
        # T greater than top of melt curve
        else:
            if P[i] <= useEOS.cp.P: phase[i]=7                  # gas
            else: phase[i] = 8                                  # supercritical fluid
    
    # print some stats:
    indmant = np.where(mat_id == mant_mat_id)
    Nmant = np.size(indmant)
    indcore = np.where(mat_id == core_mat_id)
    Ncore = np.size(indcore)
    
    indEq = np.where(np.abs(pos_z)<(zlim))
    #indEqPlanet = np.where((np.abs(pos_z[index][dataindexsort])<(zlim)) & (rrr[index][dataindexsort]<=rplanet*R_earth))
    indmantEq = np.where((np.abs(pos_z)<(zlim)) & (mat_id==mant_mat_id))
    #indmantEqPlanet = np.where((np.abs(pos_z)<(zlim)) & (rrr<=rplanet*R_earth) & (mat_id==mant_mat_id))
    indcoreEq = np.where((np.abs(pos_z)<(zlim)) & (mat_id==core_mat_id))
    #indcoreEqPlanet = np.where((np.abs(pos_z[index][dataindexsort])<(zlim)) & (rrr[index][dataindexsort]<=rplanet*R_earth) & (mat_id[index][dataindexsort]==core_mat_id))
    
    phasestrings=['Error ','Solid ','Solid + vapor ','Melt ','Liquid ','Liquid + vapor ','Vapor ','Gas ','Supercritical ']
    mantstrings=['','','','','','','','','']
    corestrings=['','','','','','','','','']
    print('Overall stats:')
    for i in range(9):
        print(phasestrings[i]+'Count: ',np.size(np.where(phase==i)),'fraction: ',np.size(np.where(phase==i))/Ntot)
    
    print('Mantle stats:')
    for i in range(9):
        print(phasestrings[i]+'Count: ',np.size(np.where(phase[indmant]==i)),'fraction: ',np.size(np.where(phase[indmant]==i))/Nmant)
        mantstrings[i]=phasestrings[i]+'(%.3g'%(100*np.size(np.where(phase[indmant]==i))/Nmant)+'%)'
    
    print('Core stats:')
    for i in range(9):
        print(phasestrings[i]+'Count: ',np.size(np.where(phase[indcore]==i)),'fraction: ',np.size(np.where(phase[indcore]==i))/Ncore)
        corestrings[i]=phasestrings[i]+'(%.3g'%(100*np.size(np.where(phase[indcore]==i))/Ncore)+'%)'
        
    print('Mantle depth stats:')
    for i in range(9):
        print(phasestrings[i]+'All min depth: ',np.amin(rrr[indmant][np.where(phase[indmant]==i)],initial=np.inf)/R_earth)
        print(phasestrings[i]+'Equatorial min depth: ',np.amin(rrr[indmantEq][np.where(phase[indmantEq]==i)],initial=np.inf)/R_earth)
    
    infernomap = plt.cm.get_cmap('inferno',8)
    cividismap = plt.cm.get_cmap('cividis',8)
    norm=colors.Normalize(vmin=1,vmax=8,clip=True)
    
    mantle=plt.scatter(rhocgs[indmant],tmp_T[indmant],c=phase[indmant],cmap=infernomap,norm=norm,s=3)
    plt.plot(MantEOS.vc.rv,MantEOS.vc.T,color='k',ls='--',lw=1)
    plt.plot(MantEOS.vc.rl,MantEOS.vc.T,color='k',ls='--',lw=1)
    plt.plot(MantEOS.mc.rl,MantEOS.mc.T,color='k',ls='--',lw=1)
    plt.plot(MantEOS.mc.rs,MantEOS.mc.T,color='k',ls='--',lw=1)
    plt.scatter(MantEOS.cp.rho,MantEOS.cp.T,color='k')
    cmant=plt.colorbar(mantle,cmap=infernomap,norm=norm)
    cmant.set_label('Phase (percent fraction)',size=28)
    cmant.ax.set_yticklabels(mantstrings[1:])
    plt.gca().set_xlim(0.95*np.amin(rhocgs[indmant]),1.05*np.amax(rhocgs[indmant]))
    plt.gca().set_ylim(0.95*np.amin(tmp_T[indmant]),1.05*np.amax(tmp_T[indmant]))
    plt.gca().set_title(MantEOS.MODELNAME+' phase diagram'+'\n'+'Simulation name: '+data.metadata.parameters.get('Snapshots:basename').decode()+'      '+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Silicate particles: %i"%(np.size(indmant)),size=20)
    plt.gca().set_xlabel(r"Density [g/cm^3]",size=25)
    plt.gca().set_ylabel(r"Temperature $[K]$",size=25)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_facecolor('grey')
    plt.gcf().set_size_inches(18,12)
    plt.tight_layout()
    imname = basename+'_Silicate_phase.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    core=plt.scatter(rhocgs[indcore],tmp_T[indcore],c=phase[indcore],cmap=infernomap,norm=norm,s=3)
    plt.plot(CoreEOS.vc.rv,CoreEOS.vc.T,color='k',ls='--',lw=1)
    plt.plot(CoreEOS.vc.rl,CoreEOS.vc.T,color='k',ls='--',lw=1)
    plt.plot(CoreEOS.mc.rl,CoreEOS.mc.T,color='k',ls='--',lw=1)
    plt.plot(CoreEOS.mc.rs,CoreEOS.mc.T,color='k',ls='--',lw=1)
    plt.scatter(CoreEOS.cp.rho,CoreEOS.cp.T,color='k')
    ccore=plt.colorbar(core,cmap=infernomap,norm=norm)
    ccore.set_label('Phase (percent fraction)',size=28)
    ccore.ax.set_yticklabels(corestrings[1:])
    plt.gca().set_xlim(0.95*np.amin(rhocgs[indcore]),1.05*np.amax(rhocgs[indcore]))
    plt.gca().set_ylim(0.95*np.amin(tmp_T[indcore]),1.05*np.amax(tmp_T[indcore]))
    plt.gca().set_title(CoreEOS.MODELNAME+' phase diagram'+'\n'+'Simulation name: '+data.metadata.parameters.get('Snapshots:basename').decode()+'      '+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Metal particles: %i"%(np.size(indcore)),size=20)
    plt.gca().set_xlabel(r"Density [g/cm^3]",size=25)
    plt.gca().set_ylabel(r"Temperature $[K]$",size=25)
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    plt.gca().set_facecolor('grey')
    plt.gcf().set_size_inches(18,12)
    plt.tight_layout()
    imname = basename+'_Metal_phase.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    mantle=plt.scatter(pos_x[indmantEq]/R_earth,pos_y[indmantEq]/R_earth,c=phase[indmantEq],cmap=infernomap,norm=norm,s=3)
    core=plt.scatter(pos_x[indcoreEq]/R_earth,pos_y[indcoreEq]/R_earth,c=phase[indcoreEq],cmap=infernomap,norm=norm,s=3)
    cmant=plt.colorbar(mantle,cmap=infernomap,norm=norm)
    #ccore=plt.colorbar(core,cmap=infernomap,norm=norm)
    cmant.ax.set_yticklabels(phasestrings[1:])
    #ccore.ax.set_yticklabels(corestrings)
    cmant.set_label('Phase',size=28)
    plt.gca().set_xlabel(r"X $[R_\oplus]$",size=20)
    plt.gca().set_ylabel(r"Y $[R_\oplus]$",size=20)
    plt.gca().set_xlim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_ylim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_aspect('equal')
    plt.gca().set_title('SPH Particle Phase Map'+'\n'+'Simulation name: '+data.metadata.parameters.get('Snapshots:basename').decode()+'      '+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Bound particles: %i"%(Ntot),size=20)
    plt.gcf().set_size_inches(18,12)
    plt.gca().set_facecolor('0.075')
    plt.tight_layout()
    imname = basename+'_Phase_map.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    mantle=plt.scatter(pos_x[indmantEq]/R_earth,pos_y[indmantEq]/R_earth,c=phase[indmantEq],cmap=infernomap,norm=norm,s=3)
    #core=plt.scatter(pos_x[indcoreEq]/R_earth,pos_y[indcoreEq]/R_earth,c=phase[indcoreEq],cmap=infernomap,norm=norm,s=3)
    cmant=plt.colorbar(mantle,cmap=infernomap,norm=norm)
    #ccore=plt.colorbar(core,cmap=infernomap,norm=norm)
    cmant.ax.set_yticklabels(mantstrings[1:])
    #ccore.ax.set_yticklabels(corestrings)
    cmant.set_label('Phase (Percent Fraction)',size=28)
    plt.gca().set_xlabel(r"X $[R_\oplus]$",size=20)
    plt.gca().set_ylabel(r"Y $[R_\oplus]$",size=20)
    plt.gca().set_xlim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_ylim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_aspect('equal')
    plt.gca().set_title('SPH Mantle Particle Phase Map'+'\n'+'Simulation name: '+data.metadata.parameters.get('Snapshots:basename').decode()+'      '+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Bound particles: %i"%(Ntot),size=20)
    plt.gcf().set_size_inches(18,12)
    plt.gca().set_facecolor('0.075')
    plt.tight_layout()
    imname = basename+'_Mantle_Phase_map.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    #mantle=plt.scatter(pos_x[indmantEq]/R_earth,pos_y[indmantEq]/R_earth,c=phase[indmantEq],cmap=infernomap,norm=norm,s=3)
    core=plt.scatter(pos_x[indcoreEq]/R_earth,pos_y[indcoreEq]/R_earth,c=phase[indcoreEq],cmap=infernomap,norm=norm,s=3)
    #cmant=plt.colorbar(mantle,cmap=infernomap,norm=norm)
    ccore=plt.colorbar(core,cmap=infernomap,norm=norm)
    #cmant.ax.set_yticklabels(mantstrings)
    ccore.ax.set_yticklabels(corestrings[1:])
    ccore.set_label('Phase (Percent Fraction)',size=28)
    plt.gca().set_xlabel(r"X $[R_\oplus]$",size=20)
    plt.gca().set_ylabel(r"Y $[R_\oplus]$",size=20)
    plt.gca().set_xlim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_ylim(-12*zlim/R_earth, 12*zlim/R_earth)
    plt.gca().set_aspect('equal')
    plt.gca().set_title('SPH Core Particle Phase Map'+'\n'+'Simulation name: '+data.metadata.parameters.get('Snapshots:basename').decode()+'      '+"Time: %06d hrs"%(snaptime/3600)+"\n"+"Bound particles: %i"%(Ntot),size=20)
    plt.gcf().set_size_inches(18,12)
    plt.gca().set_facecolor('0.075')
    plt.tight_layout()
    imname = basename+'_Core_Phase_map.png'
    plt.savefig(imname, dpi=100)
    plt.close()
    
    ind0mant=np.where(phase[indmant]==0)
    ind1mant=np.where(phase[indmant]==1)
    ind2mant=np.where(phase[indmant]==2)
    ind3mant=np.where(phase[indmant]==3)
    ind4mant=np.where(phase[indmant]==4)
    ind5mant=np.where(phase[indmant]==5)
    ind6mant=np.where(phase[indmant]==6)
    ind7mant=np.where(phase[indmant]==7)
    ind8mant=np.where(phase[indmant]==8)
    
    ind0core=np.where(phase[indcore]==0)
    ind1core=np.where(phase[indcore]==1)
    ind2core=np.where(phase[indcore]==2)
    ind3core=np.where(phase[indcore]==3)
    ind4core=np.where(phase[indcore]==4)
    ind5core=np.where(phase[indcore]==5)
    ind6core=np.where(phase[indcore]==6)
    ind7core=np.where(phase[indcore]==7)
    ind8core=np.where(phase[indcore]==8)
    
    indmants=(ind0mant,ind1mant,ind2mant,ind3mant,ind4mant,ind5mant,ind6mant,ind7mant,ind8mant)
    indcores=(ind0core,ind1core,ind2core,ind3core,ind4core,ind5core,ind6core,ind7core,ind8core)
    
    
    with open(basename+'_phase_data.txt','w') as writefile:
        writefile.write('#PHASE|Nmant|Nmant/Nmanttot|Mmant|Mmant/Mmanttot|RRRminmant|RRRminmantEq|RRRmaxmant|RRRmaxmantEq|Ncore|Ncore/Ncoretot|Mcore|Mcore/Mcoretot|RRRmincore|RRRmaxcore|'+'\n')
        for i in range(9):
            #indmanttemp=indmants[i]
            #indcoretemp=indcores[i]
            writefile.write('{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}|{:.9e}'.format(i,np.size(indmants[i]),np.size(indmants[i])/Nmant,np.sum(m[indmant][indmants[i]],dtype='float64'),np.sum(m[indmant][indmants[i]],dtype='float64')/np.sum(m[indmant],dtype='float64'),np.amin(rrr[indmant][indmants[i]],initial=np.inf),np.amin(rrr[indmantEq][np.where(phase[indmantEq]==i)],initial=np.inf),np.amax(rrr[indmant][indmants[i]],initial=-np.inf),np.amax(rrr[indmantEq][np.where(phase[indmantEq]==i)],initial=np.inf),np.size(indmants[i]),np.size(indcores[i])/Ncore,np.sum(m[indcore][indcores[i]],dtype='float64'),np.sum(m[indcore][indcores[i]],dtype='float64')/np.sum(m[indcore],dtype='float64'),np.amin(rrr[indcore][indcores[i]],initial=np.inf),np.amax(rrr[indcore][indcores[i]],initial=-np.inf))+'\n')
    print('Saved file ',basename+'_phase_data.txt')
    
    return phase
    

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
    


# ----------------------------- # 
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
    
    time = np.empty([Nsnap])
    
    if os.path.isfile(basename+'_%04d.hdf5'%Nsnap):
        snapshot = basename+'_%04d.hdf5' % Nsnap
    else:
        snapshot = basename+'_%d.hdf5' % Nsnap
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
    
    #if (os.path.isfile(basename+'_phase_data.txt')==False):
        #find_phase(databnd,bcm,bvcm,pos[indbnd],vel[indbnd],rplanet)
    #if (os.path.getmtime(basename+'_%04d.hdf5' % (Nsnap))>os.path.getmtime(basename+'_phase_data.txt')):
        #find_phase(databnd,bcm,bvcm,pos[indbnd],vel[indbnd],rplanet)
    find_phase(databnd,bcm,bvcm,pos[indbnd],vel[indbnd],rplanet)