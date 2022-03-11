import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from aco import Ant
import numpy as np
import pandas as pd
import time

model = Ant()

def measure(rho,alpha,beta,n,k,Q,ratio,maxrep,tol,optimal,rho_fixed,update):
    """
        finds optimal parameters and measures the accuracy of the different methods
    """

    #setting p value#
    if rho_fixed == True:
        p = rho
    else:
        p = -1
    
    rep = 30
    values = []
    times = []
    for i in range(0,rep,1):
        
        start = time.time()
        dist,path = model.ACO(A,p,alpha,beta,n,k,Q,random_loc=True,update=update,ratio =ratio,max_rep=20,tol = tol,log =False,plot=False,opt= optimal)
        end = time.time()
        print(dist)
        values.append(dist)
        times.append(end-start)

    print("\n===========================================================================\n")
    print("update type: ",update,"\n")
    print("\n rho : ",p,"\n")
    values = np.array(values)
    times = np.array(times)
    data = pd.DataFrame(values)
    print("Cost Information\n")
    print(data.describe())
    print("\n Time Information \n")
    data = pd.DataFrame(times)
    print(data.describe())
    print("\n===========================================================================\n")


file_link = 'Data/berlin52/berlin52.tsp'
A = model.get_distance_matrix(file_link)

#finding optimal set of parameters using simulated annealing#
# intial_param = [0.5,1,1,1000,50,1]  #[p,alpha,beta,n,k,Q]#
# param,cost = model.SA(A,intial_param,t0=1000,t1=10000,n=100,alpha=0.5,update_type='best',ratio = 0.4,tol = 0.001,rep=20)
# print("Parameters : ",param,"\n \n","total distance traveled : {}".format(cost))

param = [0.5, 1.7025269483706764, 0.1894955169216488, 1000, 76, 2.7647050477350614]
##
rho = param[0]
alpha = param[1]
beta = param[2]
n = param[3]
k = param[4]
Q = param[5]
ratio = 0.4
maxrep = 20
tol = 0.001
optimal = 7542

##GLOBAL ELITISM##
#checking rho fixed and global elitism#
measure(rho,alpha,beta,n,k,Q,ratio,maxrep,tol,optimal,rho_fixed=True,update="best")

#checking rho vary and global elitism#
measure(rho,alpha,beta,n,k,Q,ratio,maxrep,tol,optimal,rho_fixed=False,update="best")

##LOCAL ELITISM##
#checking rho fixed and local elitism#
measure(rho,alpha,beta,n,k,Q,ratio,maxrep,tol,optimal,rho_fixed=True,update="elite")

#checking rho vary and local elitism#
measure(rho,alpha,beta,n,k,Q,ratio,maxrep,tol,optimal,rho_fixed=False,update="elite")

##ALL UPDATE##
#checking rho fixed and all#
measure(rho,alpha,beta,n,k,Q,ratio,maxrep,tol,optimal,rho_fixed=True,update="all")

#checking rho vary and all#
measure(rho,alpha,beta,n,k,Q,ratio,maxrep,tol,optimal,rho_fixed=False,update="all")