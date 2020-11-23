#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:59:33 2020

@author: aMagicNeko

email: smz129@outlook.com
"""
import itertools
import scipy
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
class least_squares(object):
    def __init__(self):
        pass
    def train(self,X,y):
        """
        

        Parameters
        ----------
        X : ndarray shape:(n,m)
            data points
        y : ndarray shape:(n,)
            y of points

        Returns
        -------
        None.

        """
        n=len(X)
        tempX=np.append(X,np.ones((n,1)),axis=1)
        self.coeff=scipy.linalg.solve(np.dot(tempX.T,tempX),np.dot(tempX.T,y))
        
    def predict(self,p):
        """
        

        Parameters
        ----------
        p : ndarray shape: (m,)
            point

        Returns
        -------
        float
            

        """
        return np.dot(self.coeff,np.append(p,1))
    
    
if __name__=="__main__":
    model=least_squares()
    x1=stats.norm.rvs(size=20,loc=-2,scale=2)
    tx1=stats.norm.rvs(size=20,loc=-2,scale=2)
    y1=-1*np.ones(20)
    y2=np.ones(20)
    x2=stats.norm.rvs(size=20,loc=2,scale=2)
    tx2=stats.norm.rvs(size=20,loc=2,scale=2)
    x1=np.append(x1.reshape((-1,1)),tx1.reshape((-1,1)),axis=1)
    x2=np.append(x2.reshape((-1,1)),tx2.reshape((-1,1)),axis=1)
    x=np.append(x1,x2,axis=0)
    X=x.reshape((40,2))
    y=np.append(y1,y2)
    print(X)
    print(y)
    model.train(X,y)
    xx=np.array([*itertools.product(np.linspace(-4,4,50),np.linspace(-4,4,50))])
    is_red=lambda x:model.predict(x)<=0
    blue=np.array(list(itertools.filterfalse(is_red,xx)))
    red=np.array(list(filter(is_red,xx)))
    plt.plot(blue[:,0],blue[:,-1],'b.')
    plt.plot(red[:,0],red[:,-1],'r.')
    plt.plot(x1[:,0],x1[:,1],'r*')
    plt.plot(x2[:,0],x2[:,1],'b*')