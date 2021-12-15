from aco import Ant

model = Ant()
file_link = 'Data/eil101/eil101.tsp'
A = model.get_distance_matrix(file_link)
p = 0.5
alpha = 1
beta = 1
n = 1000
k = 75
Q = 1
dist,path = model.ACO(A,p,alpha,beta,n,k,Q,random_loc=True,update='elite',ratio =0.4,max_rep=20,tol = 0.001,log =True,plot=True,opt=629)
#model.get_graph(file_link, path,A)

print('\n============================================\n')
print('path taken \n')
print(path)
print('\npath distance was:',dist)
print('\n============================================\n')
