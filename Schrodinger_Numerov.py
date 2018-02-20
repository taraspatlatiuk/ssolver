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

steps=1000
# atomic units
hbar=1.0
m=1.0
# divide by two so a well from -W to W is of input width
W=20
# set length variable for xvec
A=W/2.0
# create x-vector from -A to A
xvec=np.linspace(-A,A,steps)
# get step size
h=xvec[1]-xvec[0]

# B matrix - to convert Numerov equation into eigenvalue problem
B = (10.0*np.diag(np.ones(steps))+np.diag(np.ones(steps-1),1)\
    +np.diag(np.ones(steps-1),-1))/(float)(12.0)
Binv = spla.inv(B)

#x0 = 0
#E,V,U = find_E(xvec,x0,steps,h,hbar,m,Binv)

n=30
# print output
#output(E,n)
# create plot
#finite_well_plot(E,V,xvec,steps,n,U)

t0 = time()
nx0 = 450
x0vec=np.linspace(-25,0,nx0)
Eall = np.zeros((steps,nx0))
for i in range(len(x0vec)):
    x0 = x0vec[i]
    Evec,Vvec,Uvec = find_E(xvec,x0,steps,h,hbar,m,Binv)
    Eall[:,i]=Evec[:]
print('Calculation time:',round(time()-t0,3),'s')

f = plt.figure()
ax=f.add_subplot(111)
for i in range(0,n):
    color=mpl.cm.jet_r((i)/(float)(n),1)
    ax.plot(x0vec,Eall[i,:],c=color)
plt.xlim(-18.0,0.0)
plt.ylim(-5.0,30.0)
plt.show()
