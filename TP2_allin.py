# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 22:25:56 2018

@author: Administrador
"""

from numpy import random, dot, zeros, shape, reshape, ones, exp,log
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import seaborn as sns
from collections import defaultdict


#-----------------------------------------SOM----------------------------------
class SOM():
    def __init__(self,entrada,arquitectura,initsigma, ilr ,Tau1, Tau2):
        
        self.entradas = entrada
        self.arquitectura = arquitectura       
        self.xdim,self.ydim = arquitectura[0:2] #dimensiones del SOM
        self.inputlen = 851
        self.sigma = initsigma
        self.learning_rate = ilr 
        self.Tau1 = Tau1
        self.Tau2 = Tau2
        self.dmap = zeros((self.xdim,self.ydim))
        print("inicializando pesos")
        self.weights = random.random(self.arquitectura)/1000000
        self.h = zeros((self.xdim,self.ydim))
        
    def decay_fun(self,x,t,T):
        return x*exp(-t/T)
     
    def dist_map(self, x):
            for i in range(self.xdim):
                for j in range(self.ydim):
                    d = x - self.weights[i,j,:]
                    self.dmap[i,j] = norm(d)
                                       
    def mapp(self, x):
        """Returns distance build map"""
        self.dist_map
        return self.dmap
    
    def neighborhood(self, win_i, sigma):
        xwin,ywin = win_i
        
        for x in range(self.xdim):
            for y in range(self.ydim):
                d_ij = sqrt((x-xwin)**2+(y-ywin)**2)
                self.h[x,y] = exp(-(d_ij**2)/(2*sigma**2))
           
    def winner(self,x):        
        self.dist_map(x)
        return np.unravel_index(np.argmin(self.dmap, axis=None), self.dmap.shape)

    def update_weights(self,x,epoch):
        
        """Updates the weights of the neurons.
        Parameters
        ----------
        x : np.array
            Current pattern to learn
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        """
        lr = self.decay_fun(self.learning_rate, epoch, self.Tau2)
        # sigma and learning rate decrease with the same rule
        sig = self.decay_fun(self.sigma, epoch, self.Tau1)
        # improves the performances
        win = self.winner(x)
        # learning_rate * neighborhood_function * (x-w)
        self.neighborhood(win, sig)  
        Ng = self.h*lr #learning_rate*neighborhood_function
        for i in range(self.xdim):
            for j in range(self.ydim):
                x_w = (x - self.weights[i,j,:])
                self.weights[i,j,:] += Ng[i,j]*(x_w) #learning_rate * neighborhood_function * (x-w) 
                #¿normalization?
        
    def train(self, epochs):
        for it in range(epochs):
            print("Epoca: ")
            print(it)
            for i in self.entradas:
                self.update_weights(i,it)
              
        print("¡Entrenada!")
                

    def win_map(self, data):
        """Returns a dictionary wm where wm[(i,j)] is a list
        with all the patterns that have been mapped in the position i,j."""
        print("Probando")
        winmap = defaultdict(list)
        c = zeros((self.xdim,self.ydim))
        X = data[0:700,1:850]
        i = 0
        for x in X:
            w = self.winner(x)
            winmap[w].append(data[i,0])
            c[w] += 1
            i += 1
        return c, winmap
    
    
def norm(x):
     return sqrt(x.dot(x.T))
 
def splittraintest(data):
        
    train=data.sample(frac=0.8,random_state = 13)
    test=data.drop(train.index)

    return train,test    
    
def centers(b,pc):
    center = []
    for k in sorted(b, key=lambda k: len(b[k]), reverse=True):
        center.append(k)
        
    return center

#-------------------------------------------Hebbian----------------------------
class Capas():
    
    def __init__(self, numero_de_neuronas,entradas_por_neurona): #entradas depende del dataset que se elija
        self.weights = random.random((entradas_por_neurona, numero_de_neuronas))/10000000
        #print(self.weights)
        
class Red():
    
    def __init__(self, capa1,alpha,rule):
        self.capa1 = capa1
        self.alpha = alpha
        self.error = []
        self.rule = rule
        
    def train(self,entradas,nsalidas,epocas):
        
        output = nsalidas
        for i in range(epocas):
            
            if(self.rule == 'Oja'):
                
                s = entradas.dot(self.capa1.weights)
                sX = s.T.dot(entradas)
                sW = (s.T.dot(s)).dot((self.capa1.weights).T)
                dw = sX - sW
                self.capa1.weights += dw.T*self.alpha
                
            if(self.rule == 'Sanger'):
                s = entradas.dot(self.capa1.weights)
                sX = s.T.dot(entradas)
                for i in range(output):
                    sW = (s[:,i].T.dot(s[:,i]))*(self.capa1.weights[:,i])
                    dw = sX[i,:] - sW
                    self.capa1.weights[:,i] += dw*self.alpha
                s = entradas.dot(self.capa1.weights)
                
#---------------------------main()---------------------------------------------
                
data = pd.read_csv('tp2_training_dataset.csv',header = None).as_matrix()
category = data[:,0]

random.seed(13)
#-----------------------------------3 Principal Components---------------------
capa1 = Capas(3,850)
Hebbian = Red(capa1,0.0001,'Sanger')

Hebbian.train(data[:,1:900], 3, 100)
w = Hebbian.capa1.weights

entrada = data[:,1:900]
trPCA = entrada.dot(w)

som = SOM(trPCA,(10,10,900),0.1,0.1,1000/log(0.1),1000)
som.train(1000)
wsom = som.weights

prueba_3PCA = np.concatenate((category,trPCA), axis = 1)
the_map,data_dict = som.win_map(prueba_3PCA)
                
#-----------------------------------9 Principal Components---------------------
capa2 = Capas(9,850)
Hebbian9 = Red(capa2,0.0001,'Sanger')
Hebbian9.train(data[:,1:900], 9, 100)

w9 = Hebbian9.capa2.weights

ninePCA = entrada.dot(w9)

som9 = SOM(ninePCA,(10,10,900),0.1,0.1,1000/log(0.1),10100)
som9.train(1000)

wsom9 = som9.train(1000)

prueba_9PCA = np.concatenate((category,ninePCA), axis = 1)
the_map9,data_dict9 = som9.win_map(ninePCA)



                