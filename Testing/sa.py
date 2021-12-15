from aco import Ant

model = Ant()
file_link = 'Data/st70/st70.tsp'
A = model.get_distance_matrix(file_link)
intial_param = [0.5,1,1,500,35,1]  #[p,alpha,beta,n,k,Q]#

param,cost,path = model.SA(A,intial_param,t0=1000,t1=10000,n=100,alpha=0.5,update_type='all',ratio = 0.3,tol = 0.001,rep=20)

print("Parameters : ",param,"\n \n","total distance traveled : {}".format(cost),"\n \n","Path taken : \n",path)

    




    