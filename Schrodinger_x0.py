import numpy as np
import scipy.linalg as spla
import matplotlib.pyplot as plt
import matplotlib as mpl
from time import time

from Schrodinger_plot import *

def harmonic_potential(x, x0):
    # get well force constant and depth
    omega = 1
    D = 0
    pot=(0.5*(omega**2)*((x-x0)**2))+D
    return pot

def diagonalize_hamiltonian(Hamiltonian):
    return spla.eigh(Hamiltonian)

def find_E(xvec,x0,steps,h,hbar,m):
    # create the potential from harmonic potential function
    U=harmonic_potential(xvec, x0)
    # create Laplacian via 3-point finite-difference method
    Laplacian=(-2.0*np.diag(np.ones(steps))+np.diag(np.ones(steps-1),1)\
        +np.diag(np.ones(steps-1),-1))/(float)(h**2)
    # create the Hamiltonian
    Hamiltonian=np.zeros((steps,steps))
    [i,j]=np.indices(Hamiltonian.shape)
    Hamiltonian[i==j]=U
    Hamiltonian+=(-0.5)*((hbar**2)/m)*Laplacian
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

n = 30
#x0 = 5
#E,V,U = find_E(xvec,x0,steps,h,hbar,m)

t0 = time()
nx0 = 150
x0vec=np.linspace(-17,0,nx0)
Eall = np.zeros((steps,nx0))
for i in range(len(x0vec)):
    x0 = x0vec[i]
    Evec,Vvec,Uvec = find_E(xvec,x0,steps,h,hbar,m)
    Eall[:,i]=Evec[:]
print('Calculation time:',round(time()-t0,3),'s')

f = plt.figure()
ax=f.add_subplot(111)
for i in range(0,n):
    color=mpl.cm.jet_r((i)/(float)(n),1)
    ax.plot(x0vec,Eall[i,:],c=color)
plt.show()

# print output
#output(E,n)
# create plot
#finite_well_plot(E,V,xvec,steps,n+1,U)
