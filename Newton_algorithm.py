

import numpy as np

#Donnée du problèmes : 

A=np.array([[55.,8.,-9.,16.,27.],\
            [8.,6.,-9.,15.,14.],\
            [-9.,-9.,22.,-24.,-31.],\
            [16.,15.,-24.,45.,34.],\
            [27.,14.,-31.,34.,75]])

# Affiche la matrice A :
#print (A)

b=np.array([[1.],\
            [-2.],\
            [-8.],\
            [7.],\
            [-8.]])

# Afficher le vecteur b : 

#print(b)

x1=np.array([[1.],[1.],[1.],[1.],[1.]])

x0=np.array([[0.],[0.],[0.],[0.],[0.]])


tol=1.e-10 # L'énoncé précise qu'il faut une tolérance à 1.e-10 près.
pas=1. # Dans le cas de la méthode de Newton simple, le pas vaut 1.
itermax=10000 # En général on prend 10 000 itérations.

#Calcul de f(x) avec f(x)=1/2 xT A x - xT b

def f(xx):
    return np.array([0.5*(xx.T.dot(A)).dot(xx)-xx.T.dot(b)])

print 'f(x1)=',f(x1)[0,0,0]
#Pour le vecteur initial [[1],[1],[1],[1],[1]] on obtient bien 152,5

#Calcul du gradient : Celui-ci s'exprime comme étant: grad f(x)= AX-b

def gradient(A,b,x):
    return (A.dot(x)-b)

#Algorithe de Newton simple:

# La Hessienne  correspond à la matrice A

def Newton_Simple(x0):
    xx=[x0] # Tous les termes sont consignés dans une liste
    dir=-(np.linalg.inv(A)).dot(gradient(A,b,xx[-1]))
    residu=[np.linalg.norm(gradient(A,b,xx[-1]),2)] # Tous les termes sont consignés dans une liste
    k=0 # Initialisation
    while residu[-1]>=tol and k<=itermax: # Condition d'arrêt
        xx.append(xx[-1]+pas*dir) #Ici, le pas vaut 1
        print(xx[-1][:]) # On affiche le vecteur qui vient d'être calculé
        dir=-np.linalg.inv(A).dot(gradient(A,b,xx[-1])) # Calcul de la nouvelle direction
        residu.append(np.linalg.norm(gradient(A,b,xx[-1]),2)) # Calcul du résidu 
        k=k+1 
    return (xx[-1],\
            k,\
            f(xx[-1])[0,0,0],\
            residu[-1]) # On affiche le vecteur final, le nombre d'itérations, la valeur de f évaluée en son minimum et le résidu

print (Newton_Simple(x0))


             # Conclusion :


print( "On constate qu'il faut une itération pour atteindre le minimum : k=1")

print( "Le vecteur final est :")
print("[0.44420182]")
print("[-5.86090488]")
print("[-1.88917518]")
print("[ 1.38176551]")
print("[-0.57980319]")


print( "Le minimum  vaut : -20.79509855 à 2.4485389275401952e-14 près")


