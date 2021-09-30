#getting imports#
import numpy as np
from ACO import AntColony

model = AntColony()

#Hyperparameter#
path = 'Data/st70/st70_tsp.txt'
A = model.get_distance_matrix_symmetric(path)
x0 = [0.62,20,10,50,45,1]
param = model.SA(A,x0,rep=5,Tmax=20,Tmin=1,epoch=10,size=70,beta_max=20,alpha_max=20,max_iter=1000,Q_max=20,log=True)

print("\n==============================\n","\n==============================\n","best set of param: \n",param)
