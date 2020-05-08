import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns
import numpy as np

from math import *
import sys

import json

#fonctions permettant de générer le fichier png du reseau de neurone:
def plot_decision_boundary(func, X, y):
    amin, bmin = X.min(axis=0) - 0.1
    amax, bmax = X.max(axis=0) + 0.1
    hticks = np.linspace(amin, amax, 101)
    vticks = np.linspace(bmin, bmax, 101)
    aa, bb = np.meshgrid(hticks, vticks)
    ab = np.c_[aa.ravel(), bb.ravel()]
    c = func(ab)
    cc = c.reshape(aa.shape)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    fig, ax = plt.subplots()
    contour = plt.contourf(aa, bb, cc, cmap=cm, alpha=0.8)
    ax_c = fig.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, 0.25, 0.5, 0.75, 1])
    
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm_bright)
    plt.xlim(amin, amax)
    plt.ylim(bmin, bmax)
    plt.title("Decision Boundary")
    #plt.show()
    fig.savefig('static/image/imageReseau.png')

def plot_histories (eta,epochs,cost_history,accuracy_history):
    fig,ax = plt.subplots(figsize=(5,5))
    ax.set_ylabel(r'$J(\theta)$')
    ax.set_xlabel('Epochs')
    ax.set_title(r"$\eta$ :{}".format(eta))
    line1, = ax.plot(range(epochs),cost_history,label='Cost')
    line2, = ax.plot(range(epochs),accuracy_history,label='Accuracy')
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})



#fonctions utiles
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - x ** 2

def relu(x):
    return np.maximum(0,x)

def relu_prime(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def leakyrelu(x):
    return np.maximum(0.01,x)

def leakyrelu_prime(x):
    x[x<=0] = 0.01
    x[x>0] = 1
    return x

def softmax(x):
    expx = np.exp(x - np.max(x))
    return expx / expx.sum(axis=0, keepdims=True)



# In[2]:


class Layer:
    def __init__(self,output,*args,**kwargs):
        
        self.output = output # Number of neurons at  layer i (current layer) 
        self.input = kwargs.get("input",None) # Number of neurons at layer i-1
        self.activ_function_curr = kwargs.get("activation",None) # Activation function for the layer
        self.parameters ={}
        self.derivatives={}
        self.activation_func=None
        if self.activ_function_curr == "relu":
            self.activation_func = relu
            self.backward_activation_func = relu_prime
        elif self.activ_function_curr == "sigmoid":
            self.activation_func = sigmoid
            self.backward_activation_func = sigmoid_prime
        elif self.activ_function_curr == "tanh":
            self.activation_func = tanh
            self.backward_activation_func = tanh_prime
        elif self.activ_function_curr == "leakyrelu":
            self.activation_func = leakyrelu
            self.backward_activation_func = leakyrelu_prime
        elif self.activ_function_curr == "softmax":
            self.activation_func = softmax
            self.backward_activation_func = softmax    
        
    def initParams(self):
        # initialisation du dictionnaire de données parameters contenant W, A et Z pour un layer
        seed=30
        np.random.seed(seed)
        self.parameters['W']=np.random.randn(self.output,self.input)*np.sqrt(2/self.input)
        self.parameters['b']=np.random.randn(self.output,1)*0.1
               
    def setW(self,matW):
        self.parameters['W']=np.copy(matW)
        
    def setA(self,matA):
        self.parameters['A']=np.copy(matA) 
        
    def setZ(self,matZ):
        self.parameters['Z']=np.copy(matZ)
    
    def setB(self,matB):
        self.parameters['b']=np.copy(matB)
        
    def setdW(self,matdW):
        self.parameters['dW']=np.copy(matdW)
        
    def setdA(self,matdA):
        self.parameters['dA']=np.copy(matdA)
        
    def setdZ(self,matdZ):
        self.parameters['dZ']=np.copy(matdZ)
    
    def setdB(self,matdB):
        self.parameters['db']=np.copy(matdB)
        


# In[31]:


class MyNetwork:
    def __init__(self):
        self.nbLayers=0
        self.layers=[]     
        
    def info(self):
        print("Content of the network:");
        j=0;
        for i in range(len(self.layers)):
            print("Layer n° ",i," => ")
            print ("\tInput ", self.layers[i].input, 
                   "\tOutput", self.layers[i].output)             
            if (i != 0):
                print ("\tActivation Function",self.layers[i].activation_func)
                print ("\tW entry layer ",i," ", self.layers[i].parameters['W'].shape,self.layers[i].parameters['W'])
                print ("\tb entry layer ",i," ", self.layers[i].parameters['b'].shape,self.layers[i].parameters['b'])

            
    def addLayer(self,layer):
        self.nbLayers += 1;
        if (self.nbLayers==1): 
            # this is the first layer so adding a layer 0
            layerZero=Layer(layer.input)
            self.layers.append(layerZero)
            
        self.layers.append(layer) 
        self.layers[self.nbLayers].input=self.layers[self.nbLayers-1].output
        self.layers[self.nbLayers].output=self.layers[self.nbLayers].output
        layer.initParams()

        
        
    def set_parametersW_b (self,numlayer,matX,matb):
        self.layers[numlayer].parameters['W']=np.copy(matX)
        self.layers[numlayer].parameters['b']=np.copy(matb)
        
        
    def forward_propagation(self, X):
        #Init predictive variables for the input layer
        self.layers[0].setA(X)
        
        #Propagation for all the layers
        for l in range(1, self.nbLayers + 1):
            # Compute Z
            self.layers[l].setZ(np.dot(self.layers[l].parameters['W'],
                                       self.layers[l-1].parameters['A'])+self.layers[l].parameters['b'])
            # Applying the activation function of the layer to Z
            self.layers[l].setA(self.layers[l].activation_func(self.layers[l].parameters['Z']))
            
    
    def cost_function(self,y):            
        return (-(y*np.log(self.layers[self.nbLayers].parameters['A']+1e-8) + (1-y)*np.log( 1 - self.layers[self.nbLayers].parameters['A']+1e-8))).mean()
    
    def backward_propagation(self,y):
        #calcul de dZ dW et db pour le dernier layer
        self.layers[self.nbLayers].derivatives['dZ']=self.layers[self.nbLayers].parameters['A']-y
        self.layers[self.nbLayers].derivatives['dW']=np.dot(self.layers[self.nbLayers].derivatives['dZ'],
                                                             np.transpose(self.layers[self.nbLayers-1].parameters['A']))
        m=self.layers[self.nbLayers].parameters['A'].shape[1]#égal au nombre de colonnes de A 
        self.layers[self.nbLayers].derivatives['db']=np.sum(self.layers[self.nbLayers].derivatives['dZ'], 
                                                       axis=1, keepdims=True) / m
        
        #calcul de dZ dW db pour les autres layers
        for l in range(self.nbLayers-1,0,-1) :
            self.layers[l].derivatives['dZ']=np.dot(np.transpose(self.layers[l+1].parameters['W']),
                                            self.layers[l+1].derivatives['dZ'])*self.layers[l].backward_activation_func(self.layers[l].parameters["Z"])
            
            self.layers[l].derivatives["dW"]=np.dot(self.layers[l].derivatives['dZ'],
                                            np.transpose(self.layers[l-1].parameters['A']))
                       
            m=self.layers[l-1].parameters['A'].shape[1]#égal au nombre de colonnes de A 
            self.layers[l].derivatives['db']=np.sum(self.layers[l].derivatives['dZ'], 
                                                       axis=1, keepdims=True) / m    
            
    def update_parameters(self, eta) :
        for l in range(1,self.nbLayers+1) :
            self.layers[l].parameters['W']-=eta*self.layers[l].derivatives['dW']
            self.layers[l].parameters["b"]-=eta*self.layers[l].derivatives["db"]
            
    def convert_prob_into_class(self,probs):
        probs = np.copy(probs)#pour ne pas perdre probs, i.e. y_hat
        probs[probs > 0.5] = 1
        probs[probs <= 0.5] = 0
        return probs
    
    def plot_W_b_epoch (self,epoch,parameter_history):
        mat=[]
        max_size_layer=0
        for l in range(1, self.nbLayers+1):    
            value=parameter_history[epoch]['W'+str(l)]
            if (parameter_history[epoch]['W'+str(l)].shape[1]>max_size_layer):
                max_size_layer=parameter_history[epoch]['W'+str(l)].shape[1]
            mat.append(value)
        figure=plt.figure(figsize=((self.nbLayers+1)*3,int (max_size_layer/2)))    
        for nb_w in range (len(mat)):    
                plt.subplot(1, len(mat), nb_w+1)
                plt.matshow(mat[nb_w],cmap = plt.cm.gist_rainbow,fignum=False, aspect='auto')
                plt.colorbar()    
        thelegend="Epoch "+str(epoch)
        plt.title (thelegend)
        #plt.show()
        

    def accuracy(self,y_hat, y):
        if self.layers[self.nbLayers].activation_func==softmax:
            # si la fonction est softmax, les valeurs sont sur différentes dimensions
            # il faut utiliser argmax avec axis=0 pour avoir un vecteur qui indique
            # où est la valeur maximale à la fois pour y_hat et pour y
            # comme cela il suffit de comparer les deux vecteurs qui indiquent 
            # dans quelle ligne se trouve le max
            y_hat_encoded=np.copy(y_hat)
            y_hat_encoded = np.argmax(y_hat_encoded, axis=0)
            y_encoded=np.copy(y)
            y_encoded=np.argmax(y_encoded, axis=0)
            return (y_hat_encoded == y_encoded).mean()
        # la dernière fonction d'activation n'est pas softmax.
        # par exemple sigmoid pour une classification binaire
        # il suffit de convertir la probabilité du résultat en classe
        y_hat_ = self.convert_prob_into_class(y_hat)
        return (y_hat_ == y).all(axis=0).mean()       
    
    def predict(self, x):
        self.forward_propagation(x)
        return self.layers[self.nbLayers].parameters['A']
    
    def next_batch(self,X, y, batchsize):
        # pour avoir X de la forme : 2 colonnes, m lignes (examples) et également y
        # cela permet de trier les 2 tableaux avec un indices de permutation       
        X=np.transpose(X)
        y=np.transpose(y)
        
        m=len(y)
        # permutation aléatoire de X et y pour faire des batchs avec des valeurs au hasard
        indices = np.random.permutation(m)
        X = X[indices]
        y = y[indices]
        for i in np.arange(0, X.shape[0], batchsize):
            # creation des batchs de taille batchsize
            yield (X[i:i + batchsize], y[i:i + batchsize])
    def fit(self, X, y ,*args,**kwargs):
        
        epochs=kwargs.get("epochs",20)
        verbose=kwargs.get("verbose",False)
        eta =kwargs.get("eta",0.01)
        batchsize=kwargs.get("batchsize",32)
    #def fit(self, X, y, epochs, eta = 0.01,batchsize=64) :
        # sauvegarde historique coût et accuracy pour affichage
        cost_history = []
        accuracy_history = []
        parameter_history = []
        for i in range(epochs):
            i+=1
            # sauvegarde des coûts et accuracy par mini-batch
            cost_batch = []
            accuracy_batch = []
            # Descente de gradient par mini-batch
            for (batchX, batchy) in self.next_batch(X, y, batchsize):
                # Extraction et traitement d'un batch à la fois
                
                # mise en place des données au bon format
                batchX=np.transpose(batchX)
                if self.layers[self.nbLayers].activation_func==softmax:
                    # la classification n'est pas binaire, y a utilisé one-hot-encoder
                    # le batchy doit donc être transposé et le résultat doit
                    # être sous la forme d'une matrice de taille batchy.shape[1]
                    
                    batchy=np.transpose(batchy.reshape((batchy.shape[0], batchy.shape[1])))
                else:
                    # il s'agit d'une classification binaire donc shape[1] n'existe
                    # pas
                    batchy=np.transpose(batchy.reshape((batchy.shape[0], 1)))
                #batchy=np.transpose(batchy.reshape((batchy.shape[0], 1)))
                self.forward_propagation(batchX)
                self.backward_propagation(batchy)
                self.update_parameters(eta)
                
                # sauvegarde pour affichage
                current_cost=self.cost_function(batchy)
                cost_batch.append(current_cost)
                y_hat = self.predict(batchX)
                current_accuracy = self.accuracy(y_hat, batchy)
                accuracy_batch.append(current_accuracy)
               
            # SaveStats on W, B as well as values for A,Z, W, b
            save_values = {}

            save_values["epoch"]=i
            for l in range(1, self.nbLayers+1):
                save_values["layer"+str(l)]=l
                # be careful A,Z,W and b must be copied otherwise it is a referencee
                #save_values['A'+str(l)]=np.copy(self.layers[l].parameters['A'])
                #save_values['Z'+str(l)]=np.copy(self.layers[l].parameters['Z'])
                a=np.copy(self.layers[l].parameters['W'])##important
                a=a.tolist() #converti numpy.array en list
                save_values['W'+str(l)]=a

                b=np.copy(self.layers[l].parameters['b']) ##dico
                b=b.tolist()
                save_values['b'+str(l)]=b ##dico

            ##envois serveur ici

            parameter_history.append(save_values)    ##tableau     
            # sauvegarde de la valeur moyenne des coûts et de l'accuracy du batch pour affichage
            current_cost=np.average(cost_batch)  
            cost_history.append(current_cost) # a récupérer
            current_accuracy=np.average(accuracy_batch)
            accuracy_history.append(current_accuracy) # a récupérer
            #affichage valeur -> plot
            
        
            if(verbose == True):
                print("Epoch : #%s/%s - %s/%s - cost : %.4f - accuracy : %.4f"%(i,epochs,X.shape[1],X.shape[1], float(current_cost), current_accuracy))
              
        return self.layers, cost_history, accuracy_history, parameter_history
        

####### MAKEMOONS ########
def makeGaussianQuantiles(epochs = 100, eta=0.01, validation_size=0.6,configNeurones=None):
    X, y = make_gaussian_quantiles(cov=3.,
                                 n_samples=1500, n_features=2,
                                 n_classes=2, random_state=1)
    parameter_history,accuracy_history,cost_history=makeNetwork(X, y, epochs, eta, validation_size, configNeurones)
    return parameter_history,accuracy_history,cost_history

#retiré car instable: les calculs plante 1 fois sur 3
def makeBlobs(epochs = 100, eta=0.01, validation_size=0.6,configNeurones=None):
    X, y = make_blobs(n_samples=1000, centers=3, n_features=2)
    parameter_history,accuracy_history,cost_history=makeNetwork(X, y, epochs, eta, validation_size, configNeurones)
    return parameter_history,accuracy_history,cost_history


# # Test sur le jeu de données MakeMoons
def makeMoons(epochs = 100, eta=0.01, validation_size=0.6,configNeurones=None):
    X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)
    parameter_history,accuracy_history,cost_history=makeNetwork(X, y, epochs, eta, validation_size, configNeurones)
    return parameter_history,accuracy_history,cost_history

def makeNetwork(X,y, epochs = 100, eta=0.01, validation_size=0.6,configNeurones=None):
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=cm_bright)#cmap=plt.cm.PiYG)
    #plt.show()

    # In[32]:


    #création d'un jeu d'apprentissage et de test

    #validation_size=0.6 #40% du jeu de données pour le test
    ##important -> paramètre a saisir via html
    

    testsize= round(1-validation_size,2)
    #print("testsize:", testsize," et valid size: ",validation_size)
    seed=30
    # séparation jeu d'apprentissage et jeu de test
    X_train,X_test,y_train,y_test=train_test_split(X, 
                                                   y, 
                                                   train_size=validation_size, 
                                                   random_state=seed,
                                                   test_size=testsize)

    # In[33]:


    #transformation des données pour être au bon format
    # X_train est de la forme : 2 colonnes, m lignes (examples)
    # y_train est de la forme : m colonnes, 1 ligne

    # La transposée de X_train est de la forme : m colonnes (exemples), 2 lignes
    X_train=np.transpose(X_train)

    # y_train est forcé pour être un tableau à 1 ligne contenant m colonnes
    y_train=np.transpose(y_train.reshape((y_train.shape[0], 1)))

    # mêmes traitements pour le jeu de test
    X_test=np.transpose(X_test)
    y_test=np.transpose(y_test.reshape((y_test.shape[0], 1)))


    # In[34]:

    
    network = MyNetwork()
    print(configNeurones)
    if configNeurones==None:
        network.addLayer(Layer(3,input=2,activation="relu"))
        network.addLayer(Layer(5,activation="relu"))
        network.addLayer(Layer(1,activation="sigmoid"))#bloquer la dernière activation
    else:
        network.addLayer(Layer(int(configNeurones[1]),input=int(configNeurones[0]),activation="relu"))
        for i in configNeurones[2:-1]:
            #print( i )
            network.addLayer(Layer(int(i),activation="relu"))
        network.addLayer(Layer(int(configNeurones[-1]),activation="sigmoid"))
        
        
        

    #network.info()


    # In[30]:


    #epochs = 100 #a saisir interface
    #eta = 0.01
    batchsize=32

    #Entraînement du classifieur
    layers,cost_history,accuracy_history,parameter_history=network.fit(X_train, y_train, verbose=False, epochs=epochs,eta=eta)


    #Prédiction
    y_pred=network.predict(X_test)
    accuracy_test = network.accuracy(y_pred, y_test)
    print("Accuracy test: %.3f"%accuracy_test)
    
    plot_histories(eta,epochs,cost_history,accuracy_history)

#Plot the decision boundary

# Affichage de la frontière de décision
    plot_decision_boundary(lambda x: network.predict(np.transpose(x)), X, y)

    return parameter_history,accuracy_history,cost_history



####### MAKECIRCLES ########


def makeCircles(epochs = 100, eta=0.01, validation_size=0.6,configNeurones=None):
    X, y = make_circles(n_samples=1000, noise=0.09, factor=0.9, random_state=0)
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=cm_bright)#cmap=plt.cm.PiYG)
    #plt.show()

    # In[36]:


    #validation_size=0.6 #40% du jeu de données pour le test

    testsize= round(1-validation_size,2)
    seed=30
    # séparation jeu d'apprentissage et jeu de test
    X_train,X_test,y_train,y_test=train_test_split(X, 
                                                   y, 
                                                   train_size=validation_size, 
                                                   random_state=seed,
                                                   test_size=testsize)


    #transformation des données pour être au bon format
    # X_train est de la forme : 2 colonnes, m lignes (examples)
    # y_train est de la forme : m colonnes, 1 ligne

    # La transposée de X_train est de la forme : m colonnes (exemples), 2 lignes
    X_train=np.transpose(X_train)

    # y_train est forcé pour être un tableau à 1 ligne contenant m colonnes
    y_train=np.transpose(y_train.reshape((y_train.shape[0], 1)))

    # mêmes traitements pour le jeu de test
    X_test=np.transpose(X_test)
    y_test=np.transpose(y_test.reshape((y_test.shape[0], 1)))


    # In[37]:


    #epochs = 100
    #eta = 0.01
    batchsize=32
    network = MyNetwork()
    print(configNeurones)
    if configNeurones==None:
        network.addLayer(Layer(25,input=2,activation="relu"))
        network.addLayer(Layer(25,activation="relu"))
        network.addLayer(Layer(3,activation="relu"))
        network.addLayer(Layer(1,activation="sigmoid"))
    else:
        network.addLayer(Layer(int(configNeurones[1]),input=int(configNeurones[0]),activation="relu"))
        for i in configNeurones[2:-1]:
            print( i )
            network.addLayer(Layer(int(i),activation="relu"))
        network.addLayer(Layer(int(configNeurones[-1]),activation="sigmoid"))
   
    #network.info()


    # In[38]:


    #Entraînement du classifieur
    layers,cost_history,accuracy_history,parameter_history=network.fit(X_train, y_train, verbose=False, epochs=epochs,eta=eta)


    #Prédiction
    y_pred=network.predict(X_test)
    accuracy_test = network.accuracy(y_pred, y_test)
    print("Accuracy test: %.3f"%accuracy_test)

    
   
    plot_histories(eta,epochs,cost_history,accuracy_history)

#Plot the decision boundary

# Affichage de la frontière de décision
    plot_decision_boundary(lambda x: network.predict(np.transpose(x)), X, y)

    return parameter_history,accuracy_history,cost_history



#makeCircles(50,0.01,0.1)
#makeMoons(50,0.01,0.1)
#makeBlobs(50,0.01,0.1) #ne pas activer, fonction instable
#makeGaussianQuantiles(50,0.01,0.1)
