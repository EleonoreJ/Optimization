# coding: cp1252
"""
TP sur la m�thode de recuit simule.

Le prob�me est celui dit du "voyageur de commerce" pour lequel un voyageur de commerce doit se rendre dans une
s�rie de ville et cherche � minimiser la distance totale parcourue.

Dans le TP propos� ici, on impose au voyageur de partir de Paris et d'y revenir.

Pour n villes � visiter, il y a donc n! parcours possibles. La fonction co�t est donc d�finie comme la
distance totale parcourue par le voyageur de commerce. Attention de bien prendre en compte le d�part et le
retour de et vers Paris.

Le travail demand� consiste � trouver l'ordre dans lequel le voyageur de commerce doit parcourir les villes
pour minimiser le trajet qu'il a � effecuter. Pour ce faire, vous utiliserez deux m�thodes :
  * une m�thode "brute force" pour laquelle vous calculerez tous les trajets possibles et s�lectionnerez le
    plus court
  * la m�thode du recuit simul�

Pour la m�thode "brute force", on pourra s'aider des fonctions suivantes :
  * itertools.permutations() (print itertools.permutations.__doc__ pour l'aide sur la fonction)
  * les m�thodes argmin() et min() des tableaux numpy
  * numpy.random.random() (print numpy.random.random.__doc__ pour l'aide sur la fonction)

Le scritp est livr� avec la classe cities dont l'instance "dico" poss�de le dictionnaire infos dont les
cl�s sont les noms de villes et tandis que les valeurs sont des dictionnaires contenant les cl�s "x" et "y"
renvoyant les coordonn�es cart�siennes des villes par rapport � Paris.

On s'attachera � comparer les temps de calcul des deux m�thodes en utilisant time.time.
"""
############################
##### IMPORTED MODULES #####
############################
import numpy as np
import time
import itertools
import matplotlib.pyplot as plt

#############################
##### CLASS DEFINITIONS #####
#############################
class cities():
    def __init__(self):
        #***** Construction du dictionnaire des villes *****
        self.infos=dict()

        self.infos['Lille']      ={'x':52.0, 'y':197.0}
        self.infos['Orl�ans']    ={'x':-33.0, 'y':-105.0}
        self.infos['Lyon']       ={'x':185.0, 'y':-343.0}
        self.infos['Paris']      ={'x':0.0, 'y':0.0}
        self.infos['Marseille']  ={'x':225.0, 'y':-617.0}
        self.infos['Strasbourg'] ={'x':403.0, 'y':-31.0}
        self.infos['Rennes']     ={'x':-300.0, 'y':-88.0}
        self.infos['Metz']       ={'x':285.0, 'y':30.0}
        self.infos['Bordeaux']   ={'x':-213.0, 'y':-448.0}
        self.infos['Perpignan']  ={'x':40.0, 'y':-688.0}
        self.infos['Cherbourg']  ={'x':-289.0, 'y':86.0}

    def ajout(self,nom,xx,yy):
        """
        Methode d'ajout d'une ville au dictionnaire.

        Entrees :
          * nom : nom de la ville, chaine de caract�re
          * xx : coordonn�e en x par rapport � Paris
          * yy : coordonn�e en y par rapport � Paris
        """
        self.infos[nom]={'x':float(xx), 'y':float(yy)}

################################
##### FUNCTION DEFINITIONS #####
################################
def cost_function(city_list, city_dict):
    """
    Fonction co�t � minimiser. Pour une liste de villes donn�e en entr�e, la fonction calcule la distance �
    parcourir pour rallier toutes ces villes une � une. Attention, la fonction prend en compte la premi�re
    distance de Paris et la derni�re distance vers Paris.

    Entr�e :
      * city_list : liste ordonn�e des villes � parcourir, liste python
      * city_dict : dictionnaire des villes contenant les informations parmettant de
                    calculer les distances � parcourir, instance de la classe cities

    Sortie :
      * la distance parcourue, float python
    """
    x=city_dict.infos[city_list[0]]['x']
    y=city_dict.infos[city_list[0]]['y']
    dist=np.sqrt(x**2+y**2)
    
    for i in range(1,len(city_list)):
        x1=city_dict.infos[city_list[i]]['x']
        y1=city_dict.infos[city_list[i]]['y']
        x0=city_dict.infos[city_list[i-1]]['x']
        y0=city_dict.infos[city_list[i-1]]['y']
        dist+=np.sqrt((x1-x0)**2+(y1-y0)**2)
        
    xn=city_dict.infos[city_list[-1]]['x']
    yn=city_dict.infos[city_list[-1]]['y']
    dist+=np.sqrt(xn**2+yn**2)
        
    return float(dist)

def compute_new_candidate(list_in):
    """
    Fonction associ�e � la m�thode de recuit simul� permettant de calculer un nouveau trajet candidat. On
    pourra utiliser la m�thode (fonction) numpy.random.permutation() pour ce faire (print
    numpy.random.permutation.__doc__ � taper dans un shell python pour avoir des informations sur
    cette fonction).

    Entr�e :
      * list_in : une liste non ordonn�e des villes � visiter par le voyageur de commerce, liste python

    Sortie :
      * une liste "al�atoire" ordonn�e des villes � visiter par le voyageur de commerce, liste python
    """
    return np.random.permutation(list_in)
    
def compute_Temp(h,k,ind,Temp):
    """
    Fonction associ�e � la m�thode de recuit simul�. Permet de calculer la nouvelles valeur de
    temp�rature � la fin d'une it�ration (voir algorithme du cours).

    Entrt�e :
      * h>0 : param�tre du calcul, float python. Plus h est petit, plus l'algorithme risque de rester
              pi�ger dans un minimum local. Plus h est grand, plus longue est la convergence de
              l'algorithme
      * k : param�tre de l'algorithme, integer python
      * ind : it�ration courante de l'algorithme, integer python
      * Temp : temp�rature courante de l'algorithme

    Sortie : 
      * nouvelle valeur du param�tre k de l'algorithme, integer python
      * nouvelle valeur de temp�rature
    """
    while np.exp((k-1)*h)>=ind or np.exp(k*h)<ind: 
        k+=1
    Temp=1.0/k
    return k,Temp

##################
##### SCRIPT #####
##################
##### Param�tres #####
#***** Dictionnaire des villes *****
dico=cities()

######################
##### QUESTION 1 #####
######################
#***** Liste non ordonn�e des villes � parcourir *****
parcours=['Marseille','Lyon','Rennes','Lille','Orl�ans','Strasbourg','Metz']

###### R�solution du probl�me en force brute #####
#print "##### R�solution du probl�me en force brute #####"
#    
##***** Calcul de toutes les permutations possibles *****
#t1=time.time()
#
#n=len(parcours)
#trajet=[]
#for i in itertools.permutations(parcours):
#    trajet.append(i)
#
##***** Calcul de la fonction co�t pour chaque permutation *****
#
#dist=[]
#for i in range (len(trajet)):
#    dist.append(cost_function(trajet[i], dico))
#distance=np.array([dist])  
#
#distmin=distance.min()
#trajetmin=trajet[distance.argmin()]
#
#print 'Nombre de trajets �tudi�s : ',len(trajet)
#
#t2=time.time()
#
#print 'Trajet le plus court :',trajetmin
#print'distance:',distmin
#print 'Temps de calcul : ',t2-t1

##### R�solution du probl�me par la m�thode du recuit simul� #####
print "##### R�solution du probl�me par la m�thode du recuit simul� #####"

#***** Param�tres du calcul *****
#----- Initialisation -----
parcours=['Marseille','Lyon','Rennes','Lille','Orl�ans','Strasbourg','Metz','Bordeaux','Perpignan','Cherbourg']
x=parcours
#----- Param�tres de l'algorithme -----
itermax=1500
hpar=1
kpar=1
Temp=1.0/kpar
Temp_list=[Temp]

#***** Algorithme de r�solution *****
t1=time.time()
for ind in xrange(2,itermax):
    #----- Calcul d'un nouveau trajet candidat -----
    ycandidat=compute_new_candidate(x)
    #----- Calcul de la diff�rence de co�t entre l'ancien et le nouveau trajet -----
    if cost_function(ycandidat, dico)<=cost_function(x, dico):
        x=ycandidat
    else: 
        delta_cost=cost_function(ycandidat, dico)-cost_function(x, dico)        
    #----- Si le nouveau trajet candidat est plus cher, il peut quand m�me -----
        u=np.random.uniform(0,1,1)  
    #----- �tre accept� avec une certaine probabilit� -----
        if np.exp(-delta_cost/Temp)>=u:
            x=ycandidat
    #----- Diminution de la temp�rature -----

    Temp_list.append(Temp)
    kpar,Temp=compute_Temp(hpar,kpar,ind,Temp)
    

t2=time.time()
#***** R�sultat *****
print 'Trajet le plus court :',x
print 'Distance :', cost_function(x, dico)
print 'Temps de calcul : ',t2-t1


#----- Profil de temp�rature -----
plt.figure()
plt.plot(Temp_list)
plt.xlabel('$n$')
plt.ylabel('$T$')
plt.title(u'Profil de temp�rature')
plt.grid()

plt.show()

######################
##### QUESTION 2 #####
######################
#***** Liste non ordonn�e des villes � parcourir *****
parcours=['Marseille','Lyon','Rennes','Lille','Orl�ans','Strasbourg','Metz','Bordeaux','Perpignan','Cherbourg']

