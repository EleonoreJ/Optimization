"""
Resolution de problemes d'optimisation avec contraintes de type egalite.
"""
############################
##### IMPORTED MODULES #####
############################
import numpy as np
import matplotlib.pyplot as plt
################################
##### FUNCTION DEFINITIONS #####
################################
#***** Probleme 1 *****
def grad_lag1(xx,mu):
    """
    Renvoie le gradient du lagrangien du probleme.

    ENTREE :
      * xx : un tableau numpy (n,1)
      * mu : un tableau numpy (m,1)

    SORTIE :
      * un tableau numpy (n,1)
    """
    x=xx[0,0]
    y=xx[1,0]

    m=mu[0,0]

    return np.array([[4*x**3+2*m*x],[4*y**3+2*m*y]])

def hess_lag1(xx,mu):
    """
    Renvoie la hessienne du lagrangien du probleme.

    ENTREE :
      * xx : un tableau numpy (n,1)
      * mu : un tableau numpy (m,1)

    SORTIE :
      * un tableau numpy (n,n)
    """
    x=xx[0,0]
    y=xx[1,0]

    m=mu[0,0]

    return np.array([[12*x**2+2*m,0.0],[0.0,12*y**2+2*m]])

def vecteur1(xx,mu):
    """
    Renvoie le vecteur a droite du probleme lineaire.

    ENTREE :
      * xx : un tableau numpy (n,1)

    SORTIE :
      * un tableau numpy (n+f,1) avec f le nombre de contraintes egalites
    """
    x=xx[0,0]
    y=xx[1,0]

    m=mu[0,0]

    return np.array([4*x**3+2*m*x,4*y**3+2*m*y,x**2+y**2-1.0])

def matrice1(xx,mu):
    """
    Renvoie la matrice a gauche du probleme lineaire.

    ENTREE :
      * xx : un tableau numpy (n,1)

    SORTIE :
      * un tableau numpy (n+f,n+f) avec f le nombre de contraintes egalites
    """
    x=xx[0,0]
    y=xx[1,0]

    m=mu[0,0]

    return np.array([[12*x**2+2*m,0.0,2*x],[0.0,12*y**2+2*m,2*y],[2*x,2*y,0.0]])

##################
##### SCRIPT #####
##################
# PROBLEME 1
print "##### PROBLEME 1 #####"
#***** Initialisation *****
#----- Liste pour les x(k) -----
xx=[np.array([[0.1],[0.1]])]

#----- Liste pour les mutltiplicateurs de lagrange -----
mu=[np.array([[0.1]])]

#----- Liste pour le residu -----
residu=[np.linalg.norm(grad_lag1(xx[-1],mu[-1]),2)]
print residu

#----- Parametres de l'algorithme -----
tol=1.0e-10
itermax=10000
k=0

#***** Boucle *****
while np.linalg.norm(grad_lag1(xx[-1],mu[-1]),2)>=tol and k<itermax:
    #----- Resolution du probleme lineaire -----
    
    
    invM=np.linalg.inv(matrice1(xx[-1],mu[-1]))
    delta=-invM.dot(vecteur1(xx[-1],mu[-1]))

    #----- Avancement des points -----
    delta_xx=np.array([[delta[0]],[delta[1]]])
    delta_mu=np.array([[delta[2]]])
    
    
    xx.append(xx[-1]+delta_xx)
    mu.append(mu[-1]+delta_mu)
    k=k+1
    
    #----- Residu -----
    residu.append(np.linalg.norm(grad_lag1(xx[-1],mu[-1]),2))
    

vp=np.linalg.eig(hess_lag1(xx[-1],mu[-1]))
plt.plot(residu)
#***** Resultats *****
print vp[0]
print xx[-1],mu[-1],np.linalg.norm(grad_lag1(xx[-1],mu[-1]),2),k

