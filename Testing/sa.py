from aco import Ant

model = Ant()
file_link = 'Data/berlin52/berlin52.tsp'
A = model.get_distance_matrix(file_link)
intial_param = [0.5,1,1,1000,50,1]  #[p,alpha,beta,n,k,Q]#

param,cost,path = model.SA(A,intial_param,t0=1000,t1=10000,n=100,alpha=0.5,update_type='best',ratio = 0.4,tol = 0.001,rep=20)

print("Parameters : ",param,"\n \n","total distance traveled : {}".format(cost))

    




    