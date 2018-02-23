import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time

from Schrodinger_plot import *

def ceo_dopping(x,x0,mag,width):
    ceo_pot = mag/width*x-mag*(1+x0/width)
    for i in range(len(ceo_pot)):
        if ceo_pot[i]>0 or ceo_pot[i]<-mag:
            ceo_pot[i]= 0
    return ceo_pot

def harmonic_potential(x,x0):
    # get well force constant and depth
    omega = 1
    D = 0
    pot=(0.5*(omega**2)*((x-x0)**2))+D
    return pot

def diagonalize_hamiltonian(Hamiltonian):
    return spla.eigh(Hamiltonian)

def find_E(xvec,x0,steps,h,hbar,m,Binv):
    # create the potential from harmonic potential function
    U=harmonic_potential(xvec, x0)+ceo_dopping(xvec, -10.0, 30.0, 2.0)
    # create Laplacian via 3-point finite-difference method
    Laplacian=(-2.0*np.diag(np.ones(steps))+np.diag(np.ones(steps-1),1)\
        +np.diag(np.ones(steps-1),-1))/(float)(h**2)
    # create the Hamiltonian
    Hamiltonian=np.zeros((steps,steps))
    [im,jm]=np.indices(Hamiltonian.shape)
    Hamiltonian[im==jm]=U
    Hamiltonian+=(-0.5)*((hbar**2)/m)*Binv.dot(Laplacian)
    # diagonalize the Hamiltonian yielding the wavefunctions and energies
    E,V=diagonalize_hamiltonian(Hamiltonian)
    # determine theoretical number of energy levels (n)
    return (E,V,U)

def cakc_E_psi(nx0):
    steps=1000
    hbar=1.0
    m=1.0
    W=20
    A=W/2.0
    xvec=np.linspace(-A,A,steps)
    # get step size
    h=xvec[1]-xvec[0]

    # B matrix - to convert Numerov equation into eigenvalue problem
    B = (10.0*np.diag(np.ones(steps))+np.diag(np.ones(steps-1),1)\
        +np.diag(np.ones(steps-1),-1))/(float)(12.0)
    Binv = spla.inv(B)

    #x0 = 0
    #E,V,U = find_E(xvec,x0,steps,h,hbar,m,Binv)

    # print output
    #output(E,n)
    # create plot
    #finite_well_plot(E,V,xvec,steps,n,U)

    t0 = time()
    x0vec=np.linspace(-20,5,nx0)
    Eall = np.zeros((steps,nx0))

    psiall = np.zeros((len(xvec),steps,len(x0vec)))
    for i in range(len(x0vec)):
        x0 = x0vec[i]
        Evec,Vvec,Uvec = find_E(xvec,x0,steps,h,hbar,m,Binv)
        psiall[:,:,i] = Vvec[:,:]**2
        Eall[:,i]=Evec[:]
    print('Calculation time:',round(time()-t0,3),'s')

    return (Eall,psiall,xvec,x0vec)

#def density(Eall, psiall, xvec, x0vec):

# number of modes = 5
xvec = np.linspace(-2,2,20)
x0vec = np.linspace(-1,1,10)
Eall = np.zeros((5,10))
psiall = np.zeros((5,20,10))

for i, x in enumerate(x0vec):
    Eall[:,i] = x0vec**2




























a = 1
