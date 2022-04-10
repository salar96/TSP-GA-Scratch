#!/usr/bin/env python
# coding: utf-8

# # Solving TSP with Genetic Algorithm
# Created on Thu Jun 13 18:08:36 2019 
# @author: Salar Basiri - 97200346 M.Sc Mechatronics @ SUT
# salarbsr.1996@gmail.com
# Genetic Algorithm
# HW4 - Intelligent Systems
# Dr.Broushaki
# """

# # Introduction
# in this homework, we are about to solve the Traveling Salesman Problem with Genetic Algorithm and find the shortest route that covers all of the 14 cities (city num 0 also included) and returns to the start point.
# we know by intuition that the shortest sequence of cities is the one that starts from 0 and ends with the last city in ascending order, but we're about to find if GA Algorithm can also get that result if we start it with some random sequences or not.
# at first, some libraries added and after that, the process begins.
# code is completely commented for better readability, no further information need to be included.

# In[1]:


#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random
from random import choices
import math
from math import pow
from collections import Counter


# In[6]:


#%% Initial parameters
N_ipop=800; # number of initial population
myu=0.04; # mutation ratio
city_num=13; # number of total cities
mate_gen_num=2; # number of gens to be replaced in reproduction
max_iter=50; # maximum number of iterations
repeat_num=5; # number of times that the algorithm starts from the begining
global_min_cost=1e6; # the final cost that we are about to report
global_best_seq=np.zeros(city_num+2).reshape(1,city_num+2) #the final sequence that we are about to report
for repeat in range(1,repeat_num+1):  # this is the main outside loop
    print("*********************************************************************************")
    print("*********************************************************************************")
    print("*********************************************************************************")
       #%% Generating initial population
        
    city=[1,2,3,4,5,6,7,8,9,10,11,12,13]
    ipop=np.random.rand(N_ipop,city_num)
    for i in range(0,N_ipop):
        ipop[i,:]=random.sample(city,len(city))
    print(ipop)    
    z=np.zeros(N_ipop);
    zeezee=z.reshape(N_ipop,1)
    ipop=np.concatenate((zeezee,ipop),axis=1)
    ipop=np.concatenate((ipop,zeezee),axis=1)
        #%% Natural Selection function
    def give_coor(a):
        # this is function returns coordinates of a selected city a:
        if a<=5:
            y=0 
            x=a
        elif (a==6):
            y=1
            x=5
        elif a==13:
            x=0
            y=1
        else:
            y=2
            x=np.abs(a-12)    
        return [x,y]
            

           
    ###########################################################

    def cost(x):
        # this function calculates the total distance between cities in a vector x:
        s1=x.shape[0]
        s2=x.shape[1]
        C=np.zeros(s1)
        for i in range(0,s1):
            C[i]=0
            for j in range(0,s2-1):
               C[i]=C[i]+math.sqrt(pow((give_coor(x[i,j])[0]- give_coor(x[i,j+1])[0]),2)+ pow((give_coor(x[i,j])[1]- give_coor(x[i,j+1])[1]),2))
        return C
    
    
        #%% here we sort our initial population based on their cost
    ipop_eval=cost(ipop)
    I=np.argsort(ipop_eval,kind='stable')
        #%%
    npop=np.random.rand(ipop.shape[0],ipop.shape[1])
    for n in range (0,N_ipop):
        k=I[n]
        npop[n,:]=ipop[k,:]
     
         #%% Deleting one half of ipop
    npop=npop[:int(N_ipop/2)]
    #%% and good half on the rest
    N=int(npop.shape[0]/2)
    pop_good=npop[:N]
    def check_repeat(x):    
        check=Counter(x[0])
        h=-1
        for i in range(1,city_num+1):
            if check[i]>1:
                h=i
        return h    
                
    
    def mate(x,y):
        # this function generates two children with given parents x , y
        u=x.copy()
        v=y.copy()
        r=int(np.random.rand()*(x.shape[1]-3))+1
        a=[u[0,r],u[0,r+1]]
        b=[v[0,r],v[0,r+1]]
        [u[0,r],u[0,r+1]]=b
        [v[0,r],v[0,r+1]]=a
        t1=0
        t2=0
        d=-10
        c=-10
        while check_repeat(u)>0 or check_repeat(v)>0:
            for i in range(0,u.shape[1]):
                if u[0,i]==check_repeat(u):
                    t1=i
                    c=u[0,i]
            for j in range(0,u.shape[1]):
                if v[0,j]==check_repeat(v):
                    t2=j 
                    d=v[0,j]
                
            u[0,t1]=d
            v[0,t2]=c
        return u,v
         
    
    #%% Reproducing childs
    mean_cost=np.zeros(50)
    final_cost=np.zeros(50)
    for GA_iter in range(1,max_iter):

        p=np.array(range(0,N))
        weights=np.zeros(N)
        for i in range(1,N+1):
            weights[i-1]=(N-i+1)/sum(p)
            weights[i-1]=weights[i-1]+1/(15*i)*0.01
            if i<10:
                weights[i-1]=weights[i-1]*4;

        pop_child=pop_good.copy()
          
        for i in range(0,int(N),2):
            r1=choices(p,weights)   
            r2=choices(p,weights) 
            pop_child[i,:],pop_child[i+1,:]=mate(pop_good[r1,:].reshape(1,city_num+2),pop_good[r2,:].reshape(1,city_num+2))
            
        new_pop=np.concatenate((pop_good,pop_child))
        
        
        #% mutation
        myu_num=int(myu*new_pop.shape[0])
        
        for i in range(1,myu_num+1):
            r1=int(np.random.rand()*(new_pop.shape[0]-1))+1
            r2=int(np.random.rand()*(new_pop.shape[1]-3))+1
            r3=int(np.random.rand()*(new_pop.shape[1]-3))+1
            a=new_pop[r1,r2]
            b=new_pop[r1,r3]
            new_pop[r1,r2]=b
            new_pop[r1,r3]=a
        
        eeval=cost(new_pop)
        mean_cost[GA_iter-1]=np.mean(eeval)
        I=np.argsort(eeval,kind='stable')
        npop=np.random.rand(new_pop.shape[0],new_pop.shape[1])
        for n in range (0,new_pop.shape[0]):
            k=I[n]
            npop[n,:]=new_pop[k,:]
        new_pop=npop.copy()
        pop_good=new_pop[:N]
        final=pop_good[0,:]
        final_cost[GA_iter-1]=cost(final.reshape(1,city_num+2))
        print("repeat num:",repeat,"iter #",GA_iter,":","Min cost is:" , final_cost[GA_iter-1],"Mean Cost is:",mean_cost[GA_iter-1])
       
        if final_cost[GA_iter-1] < global_min_cost:
            global_min_cost=final_cost[GA_iter-1]
            global_best_seq=final
    figure(num=None, figsize=(6, 10), dpi=80, facecolor='w', edgecolor='k')       
    plt.subplot(211)
    plt.plot(mean_cost[:-1],linewidth=2,color='r')
    plt.ylabel("Mean cost")
    plt.xlabel("Iteration")
    ptit="Mean cost diagram for repeat # "+str(repeat)
    plt.title(ptit)
    plt.subplot(212)
    plt.plot(final_cost[:-1],linewidth=2,color='b')
    plt.ylabel("Min cost")
    plt.xlabel("Iteration")
    ptit="Min cost diagram for repeat # "+str(repeat)
    plt.title(ptit)
    plt.show()
print("*********************************************************************************")
print("*********************************************************************************")
print("*********************************************************************************")    
print("global best cost is:" , global_min_cost)
print("global best sequence is:" , global_best_seq)
      
              


# # Conclusion
# 
# as the results suggest, the final sequence reported is the one that was expected with cost 14 which is the global minimum so the algorithm is working correctly.
# we set up 5 repeats to start from a different initial population in order to avoid local minimums and get global result. each repeat includes 50 iterations so the total computation time is very little.
# diagrams above suggest that both min cost and mean cost have descending trends in general, but it may be useful to notice that because of the mutation modeling inside the process, some small fluctuations can be seen in all mean cost diagrams, but descending trend in min cost diagrams is much more stable and no ascending fluctuations can be seen. both diagrams in all repeats are very similar to diagrams shown in ((figure 6-2)) of the book.
# 
# good luck!
