# -*- coding: cp1252 -*-
"""
Methodes de gradients.
"""
############################
##### IMPORTED MODULES #####
############################
import numpy as np
import matplotlib.pyplot as plt
import itertools
import cPickle
import scipy.linalg as spl

################################
##### FUNCTION DEFINITIONS #####
################################
def function(A,b,c,xx):
    shape=xx[0].shape
    ZZ=np.zeros_like(xx[0])
    for item in itertools.product(*map(xrange,shape)):
        vect=list()
        for elem in xx:
            vect.append(elem[item])
        vec=np.array(vect,ndmin=1)
        ZZ[item]=0.5*np.dot(vec.T,A.dot(vec))-np.dot(vec.T,b) + c
    return ZZ

def gradient(A,b,x):
    return A.dot(x)-b

def pas_fixe(x0,fonction,pas=1.0e-2,tol=1.0e-10,itermax=10000):
    A=fonction['A']
    b=fonction['b']
    c=fonction['c']

    #***** Initialisation *****
    xx=[x0]
    dir= - gradient(A,b,xx[-1])
    residu=[np.linalg.norm(gradient(A,b,xx[-1]))]

    k=0
    while residu[-1] > tol and k < itermax :
        #----- Calcul de x(k+1) -----
        xx.append(xx[-1]+dir*pas)

        #----- Calcul de la nouvelle direction de descente d(x+1) -----
        dir = - gradient(A,b,xx[-1])

        #----- Calcul du residu r(k+1) -----
        residu.append(np.linalg.norm(gradient(A,b,xx[-1])))
        k += 1

    return {'xx':np.asarray(xx),'residu':np.asarray(residu)}

def conjugate(x0,fonction,tol=1.0e-10,itermax=10000):
    A=fonction['A']
    b=fonction['b']
    c=fonction['c']

    #***** Initialisation *****
    xx=[x0]
    dir = - gradient(A,b,xx[-1])
    p = dir
    residu = [np.linalg.norm(gradient(A,b,xx[-1]))]

    k=0
    while residu[-1] > tol and k < itermax :
        #----- Calcul de rho(k) -----
        rho = (np.linalg.norm(dir))**2/(np.transpose(p).dot(A).dot(p))
        
        #----- Calcul de x(k+1) -----
        xx.append(xx[-1]+p*rho)

        #----- Calcul de la direction d(k+1) -----
        dir = dir-rho*A.dot(p)

        #----- Calcul de beta(k) -----
        beta=np.linalg.norm(dir)**2 / residu[-1]**2
        
        #----- Calcul du projeté p(x+1) -----
        p = dir+beta*p

        #----- Calcul du résidu r(x+1) -----
        residu.append(np.linalg.norm(dir))
        
        k += 1

    return {'xx':np.asarray(xx),'residu':np.asarray(residu)}

def plot(xx,res):
    plt.figure()
    
    X1, X2 = np.meshgrid(np.linspace(-5.0,5.0,101),np.linspace(-5.0,5.0,101))
    Z=function(A,b,c,[X1,X2])
    
    plt.contour(X1,X2,Z)

    plt.plot(xx.T[0],xx.T[1],'k-x')

    plt.axes().set_aspect('equal')
    
    plt.figure()
    plt.plot(res)
    plt.yscale('log')
    plt.grid()
    plt.xlabel('Iterations')
    plt.ylabel(r'$||\nabla f(x) ||_2$')
    plt.title('Convergence')

    plt.show()

#######################
##### SCRIPT PART #####
#######################

###############################
##### SELF-SUSTAINED PART #####
###############################
if __name__=="__main__":
    #***** Exercice 01 *****
    """
    Pour cet exercice le minimum est [-0.1429,-0.4286]
    """
    x0=np.array([3,3])
    A=np.array([[4,1],[1,2]])
    b=np.array([-1,-1])
    c=np.zeros(1)

    fonction={'A':A, 'b':b, 'c':c}

    #----- Pas fixe -----
    cas_01=pas_fixe(x0,fonction,1.0e-1,1.0e-6,10000)
    print cas_01['xx'][-1], len(cas_01['xx'])-1

    plot(cas_01['xx'],cas_01['residu'])

    #----- Gradient conjugue -----
    cas_02=conjugate(x0,fonction,1.0e-6,10000)
    print cas_02['xx'][-1], len(cas_02['xx'])-1

    plot(cas_02['xx'],cas_02['residu'])

    #***** Exercice 02 *****

    #***** Exercice 03 *****
    #----- Import data -----
    """
    Les donnees sont stockees dans la liste data sous forme de dictionnaire ayant les cles 'A' et
    'b'
    """
    data=list()
    for ind in xrange(1,6):
        data.append(cPickle.load(open('paire_%i.pickle' % ind,'r')))
        data[ind-1]['A']=np.asarray(data[ind-1]['A'].todense())

    
