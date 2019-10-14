import numpy as np
import itertools

#METHODE DE GRADIENT A PAS FIXE

#Donnees de base

x0=np.array([[0],[0],[0],[0]])
M=np.array([[1.0,0.31,0.0961,0.029791],[1.0,1.57,2.4649,3.869893],[1.0,3.04,9.2416,28.094464],[1.0,3.49,12.1801,42.508549],[1.0,4.45,19.8025,88.121125]])
y=np.array([[4.10],[4.70],[2.02],[-2.79],[-2.58]])


#Calcul des matrices

A=(M.T).dot(M)
b=(M.T).dot(y)
c=0.5*(y.T).dot(y)

#Definition fonction

def fonction(A,b,c,xx):
    shape=xx[0].shape
    F=np.zeros_like(xx[0])
    for item in itertools.product(*map(xrange,shape)):
        vect=list()
        for elem in xx:
            vect.append(elem[item])
        vec=np.array(vect,ndim=1)
        F[item]=0.5*np.dot(vec.T,A.dot(vec))-np.dot(vec.T,b)+c
    return F

#Calcul de gradient          

def gradient(A,b,x):
    return A.dot(x)-b

#Methode de gradient a pas fixe

def pas_fixe(x0,fonction,pas=1.0e-4,tol=1.0e-2,itermax=1000000):

#Initialisation

    xx=[x0]
    dir=-gradient(A,b,xx[-1])
    residu=[np.linalg.norm(gradient(A,b,xx[-1]))]
    k=0

#Boucle

    while residu[-1]>tol and k<itermax:
        xx.append(xx[-1]+dir*pas)
        dir=-gradient(A,b,xx[-1])

        residu.append(np.linalg.norm(gradient(A,b,xx[-1])))

        k+=1
    print 'Le resultat est obtenu en',k,'iterations.'

    amin=xx[-1]

    print 'La vecteur amin obtenu est',amin
    
    f=0.5*amin.T.dot(A).dot(amin)-amin.T.dot(b)+c

    print 'La valeur de f(amin) est :',f

    print "Ce qui correspond bien à l'indication fournie dans l'énoncé."

    print 'Nous affichons maintenant les résidus et les vecteurs a successifs'

    return {'xx':np.asarray(xx),'residu':np.asarray(residu)}

print pas_fixe(x0,fonction)



    
