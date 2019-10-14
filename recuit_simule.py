# coding: cp1252
"""
TP sur la méthode de recuit simule.

Le probème est celui dit du "voyageur de commerce" pour lequel un voyageur de commerce doit se rendre dans une
série de ville et cherche à minimiser la distance totale parcourue.

Dans le TP proposé ici, on impose au voyageur de partir de Paris et d'y revenir.

Pour n villes à visiter, il y a donc n! parcours possibles. La fonction coût est donc définie comme la
distance totale parcourue par le voyageur de commerce. Attention de bien prendre en compte le départ et le
retour de et vers Paris.

Le travail demandé consiste à trouver l'ordre dans lequel le voyageur de commerce doit parcourir les villes
pour minimiser le trajet qu'il a à effecuter. Pour ce faire, vous utiliserez deux méthodes :
  * une méthode "brute force" pour laquelle vous calculerez tous les trajets possibles et sélectionnerez le
    plus court
  * la méthode du recuit simulé

Pour la méthode "brute force", on pourra s'aider des fonctions suivantes :
  * itertools.permutations() (print itertools.permutations.__doc__ pour l'aide sur la fonction)
  * les méthodes argmin() et min() des tableaux numpy
  * numpy.random.random() (print numpy.random.random.__doc__ pour l'aide sur la fonction)

Le scritp est livré avec la classe cities dont l'instance "dico" possède le dictionnaire infos dont les
clés sont les noms de villes et tandis que les valeurs sont des dictionnaires contenant les clés "x" et "y"
renvoyant les coordonnées cartésiennes des villes par rapport à Paris.

On s'attachera à comparer les temps de calcul des deux méthodes en utilisant time.time.
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
        self.infos['Orléans']    ={'x':-33.0, 'y':-105.0}
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
          * nom : nom de la ville, chaine de caractère
          * xx : coordonnée en x par rapport à Paris
          * yy : coordonnée en y par rapport à Paris
        """
        self.infos[nom]={'x':float(xx), 'y':float(yy)}

################################
##### FUNCTION DEFINITIONS #####
################################
def cost_function(city_list, city_dict):
    """
    Fonction coût à minimiser. Pour une liste de villes donnée en entrée, la fonction calcule la distance à
    parcourir pour rallier toutes ces villes une à une. Attention, la fonction prend en compte la première
    distance de Paris et la dernière distance vers Paris.

    Entrée :
      * city_list : liste ordonnée des villes à parcourir, liste python
      * city_dict : dictionnaire des villes contenant les informations parmettant de
                    calculer les distances à parcourir, instance de la classe cities

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
    Fonction associée à la méthode de recuit simulé permettant de calculer un nouveau trajet candidat. On
    pourra utiliser la méthode (fonction) numpy.random.permutation() pour ce faire (print
    numpy.random.permutation.__doc__ à taper dans un shell python pour avoir des informations sur
    cette fonction).

    Entrée :
      * list_in : une liste non ordonnée des villes à visiter par le voyageur de commerce, liste python

    Sortie :
      * une liste "aléatoire" ordonnée des villes à visiter par le voyageur de commerce, liste python
    """
    return np.random.permutation(list_in)
    
def compute_Temp(h,k,ind,Temp):
    """
    Fonction associée à la méthode de recuit simulé. Permet de calculer la nouvelles valeur de
    température à la fin d'une itération (voir algorithme du cours).

    Entrtée :
      * h>0 : paramètre du calcul, float python. Plus h est petit, plus l'algorithme risque de rester
              piéger dans un minimum local. Plus h est grand, plus longue est la convergence de
              l'algorithme
      * k : paramètre de l'algorithme, integer python
      * ind : itération courante de l'algorithme, integer python
      * Temp : température courante de l'algorithme

    Sortie : 
      * nouvelle valeur du paramètre k de l'algorithme, integer python
      * nouvelle valeur de température
    """
    while np.exp((k-1)*h)>=ind or np.exp(k*h)<ind: 
        k+=1
    Temp=1.0/k
    return k,Temp

##################
##### SCRIPT #####
##################
##### Paramètres #####
#***** Dictionnaire des villes *****
dico=cities()

######################
##### QUESTION 1 #####
######################
#***** Liste non ordonnée des villes à parcourir *****
parcours=['Marseille','Lyon','Rennes','Lille','Orléans','Strasbourg','Metz']

###### Résolution du problème en force brute #####
#print "##### Résolution du problème en force brute #####"
#    
##***** Calcul de toutes les permutations possibles *****
#t1=time.time()
#
#n=len(parcours)
#trajet=[]
#for i in itertools.permutations(parcours):
#    trajet.append(i)
#
##***** Calcul de la fonction coût pour chaque permutation *****
#
#dist=[]
#for i in range (len(trajet)):
#    dist.append(cost_function(trajet[i], dico))
#distance=np.array([dist])  
#
#distmin=distance.min()
#trajetmin=trajet[distance.argmin()]
#
#print 'Nombre de trajets étudiés : ',len(trajet)
#
#t2=time.time()
#
#print 'Trajet le plus court :',trajetmin
#print'distance:',distmin
#print 'Temps de calcul : ',t2-t1

##### Résolution du problème par la méthode du recuit simulé #####
print "##### Résolution du problème par la méthode du recuit simulé #####"

#***** Paramètres du calcul *****
#----- Initialisation -----
parcours=['Marseille','Lyon','Rennes','Lille','Orléans','Strasbourg','Metz','Bordeaux','Perpignan','Cherbourg']
x=parcours
#----- Paramètres de l'algorithme -----
itermax=1500
hpar=1
kpar=1
Temp=1.0/kpar
Temp_list=[Temp]

#***** Algorithme de résolution *****
t1=time.time()
for ind in xrange(2,itermax):
    #----- Calcul d'un nouveau trajet candidat -----
    ycandidat=compute_new_candidate(x)
    #----- Calcul de la différence de coût entre l'ancien et le nouveau trajet -----
    if cost_function(ycandidat, dico)<=cost_function(x, dico):
        x=ycandidat
    else: 
        delta_cost=cost_function(ycandidat, dico)-cost_function(x, dico)        
    #----- Si le nouveau trajet candidat est plus cher, il peut quand même -----
        u=np.random.uniform(0,1,1)  
    #----- être accepté avec une certaine probabilité -----
        if np.exp(-delta_cost/Temp)>=u:
            x=ycandidat
    #----- Diminution de la température -----

    Temp_list.append(Temp)
    kpar,Temp=compute_Temp(hpar,kpar,ind,Temp)
    

t2=time.time()
#***** Résultat *****
print 'Trajet le plus court :',x
print 'Distance :', cost_function(x, dico)
print 'Temps de calcul : ',t2-t1


#----- Profil de température -----
plt.figure()
plt.plot(Temp_list)
plt.xlabel('$n$')
plt.ylabel('$T$')
plt.title(u'Profil de température')
plt.grid()

plt.show()

######################
##### QUESTION 2 #####
######################
#***** Liste non ordonnée des villes à parcourir *****
parcours=['Marseille','Lyon','Rennes','Lille','Orléans','Strasbourg','Metz','Bordeaux','Perpignan','Cherbourg']

