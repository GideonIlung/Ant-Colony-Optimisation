#Imports
import numpy as np
import tsplib95
import networkx
import warnings
warnings.filterwarnings("ignore")
#np.seterr(all='raise')
import random
import pandas as pd
from scipy.spatial import distance

class AntColony:
    def __init__(self):
        print("loaded")

#############################################################################################################################
    def getProbabilityMatrix(self,A,P,alpha,beta):
        """
            Generates the probability matrix based on the pheromone and distance matrix

            Inputs
                A     : the distance matrix
                P     : the Pheromone Matrix
                alpha : the Pheromone weight value
                beta  : the distance weight value

            Outputs
                X     : the probability matrix
        """

        ########METHOD 1##########################
        # C = 1/A
        # X = (P**alpha) * (C ** beta)

        # n = len(X)

        # for i in range(n):
        #     deno = sum(X[i,:])
        #     X[i,:] = (1/deno) * X[i,:]
        
        # return X
        ##########################################

        #####METHOD 2#############################
        n = len(A)

        X = np.zeros((n,n))

        for i in range(0,n,1):

            #calculating the total weight across path
            deno = 0

            for j in range(0,n,1):

                if(A[i,j]!=0):
                    deno += (P[i,j]**alpha)*((1/A[i,j])**beta)

            #updating probability matrix
            for j in range(0,n,1):
                
                if i!=j and A[i,j]!=0 and A[i,j]!=np.inf:
                    X[i,j] = (P[i,j]**alpha)*((1/A[i,j])**beta)/deno
                else:
                    X[i,j] = 0
        
        return X
  ####################################################################################################################  
    def accumulator(self,v):
        """
            Creates the accumulator vector from the input given
        """

        u = []

        for i in range(len(v)):
            temp = sum(v[i:])
            u.append(temp)

        r = random.uniform(0,u[0])
        index = -1

        for i in range(0,len(u),1):
            
            if i!=len(u)-1:
                if (r<=u[i]) and (r>u[i+1]):
                    index = i
                    break
            else:
                if (r>0) and (r<=u[-1]):
                    index = i

        return index
######################################################################################################################
    def ant_path(self,A,P,Q,start):
        """
            Inputs:
                A     : distance matrix
                P     : probability matrix
                Q     : pheromone update parameter
                start : index of starting location of agent

            Outputs:
                dist  : distance traveled by agent
                path  : path taken by agent

                C     : pheromone update trail
        """

        n = len(A)
        X = np.copy(P)
        C = np.zeros((n,n))
        dist = 0
        path = [start]
        i = start
        count = 0
        valid = True

        while count < n-1:

            #getting the row of probabilities corresponding current node#
            v = X[i,:]

            #getting next node to visit#
            j = self.accumulator(v)

            #checks if solution is valid#
            if(j==-1):
                valid = False
                break

            dist+= A[i,j]
            path.append(j)

            #making sure agent doesnt visit node twice#
            X[:,i] = 0

            i = j
            count+=1
        

        dist+= A[i,start]
        path.append(start)

        for j in range(len(path)-1):
            row = path[j]
            col = path[j+1]
            C[row,col] += Q/dist
        

        return dist,path,C,valid
######################################################################################################################
    def reorder_path(self,path,start):
        """
            reorders path starting and ending with start index

            Input:
                path     : path which is a permuatation that will be reordered
                start    : the starting index 

            outputs:
                new_path : reordered path
        """

        index = path.index(start)

        if index!=0:
            new_path = []

            for i in range(index,len(path),1):
                new_path.append(path[i])
            
            for i in range(1,index,1):
                new_path.append(path[i])
            
            new_path.append(start)
            return new_path
        else:
            return path
########################################################################################################################
    def updatePheromone(self,p,X,C,i,n):
        """
            updates the pheromone matrix depending on parameters entered
        """

        s = p

        if p == -1:
            s = 1 - (i+1)/n
        
        ans = (1-s)*X + C
        
        return ans
#########################################################################################################################
    def sort_elite(self,elite_dist,elite_C):
        """
            sorts the elite lists
        """
        n = len(elite_dist)
        dist = elite_dist.copy()
        C = elite_C.copy()
        
        for i in range(0,n,1):
            for j in range(i+1,n,1):
                
                if elite_dist[j]<elite_dist[i]:
                    
                    #sorting distances#
                    temp_dist = dist[i]
                    dist[i] = dist[j]
                    dist[j] = temp_dist
                    
                    #sorting pheromone matrix#
                    temp_C = C[i]
                    C[i] = C[j]
                    C[j] = temp_C
        
        return dist,C
###########################################################################################################################
    def update_Elite(self,elite_dist,elite_C,ratio,k,temp_dist,temp_C):
        """
            updates the list of elite agents that will be used to update the trail for the following agents
        """
        
        size = round(ratio*k)
        dist = elite_dist.copy()
        C = elite_C.copy()
        
        dist.append(temp_dist)
        C.append(temp_C)
        
        new_dist,new_C = sort_elite(dist,C)
        
        while len(new_dist)>size:
            new_dist.pop()
            new_C.pop()
        
        return new_dist,new_C
###########################################################################################################################
    def get_elite_C(self,elite_C):
        """
            sums up the elite matrices into 1 matrix
        """
        n = len(elite_C)
        shape = len(elite_C[0])
        
        new_C = np.zeros((shape,shape))
        
        for i in range(n):
            new_C += elite_C[i]
        
        return new_C
##########################################################################################################################
    def ACO(self,A,p,alpha,beta,n,k,Q,random_loc,update,ratio,max_rep,tol,log):
        """
        Finds optimal route using Ant Colony Optimisation techniques
        
        Inputs:
            A           : Distance Matrix
            p           : (scalar) evaporation rate
            alpha       : (scalar) parameter that affects pheromone weighting
            beta        : (scalar) parameter that affects distance weighting
            n           : (scalar) number of interations to be performed
            k           : (scalar) number of ants to be used
            random_loc  : boolean variable that assigns agents to random nodes or not
            update      : how the pheromone matrix is updated (all: all agents pheromone is updated) ; (best: best agent updates pheromone matrix)
            ratio       : percentage of the top solution to select from
            max_rep     : how times same solutions pop up before terminating 
            tol         : the tolerance solutions generated
            
        Output:
            dist        : the distance traveled by the last agent
            path        : set of 2-tuples of route to be taken
        """

        #Pheromone Matrix#
        X = np.ones((len(A),len(A)))

        #intialisation#
        P = self.getProbabilityMatrix(A,X, alpha, beta)
        start = 0

        if random_loc == True:
            start = np.random.randint(len(A))


        valid = False

        a_rep = 0
        while valid == False and a_rep<max_rep:
            best_dist,best_path,best_C,valid = self.ant_path(A,P,Q,start)
            a_rep+=1
        
        if a_rep >=max_rep:
            print("Error")
            return 0,0

        rep = 0
        
        
        #elitism#
        elite_dist = [best_dist]
        elite_C = [best_C]

        for i in range(n):

            C_all = np.zeros((len(A),len(A)))

            #getting probability matrix#
            P = self.getProbabilityMatrix(A,X, alpha, beta)

            #constructing ant paths#
            valid = False
            
            a_rep = 0
            while valid == False and a_rep<max_rep:
                iter_dist,iter_path,iter_C,valid = self.ant_path(A,P,Q,start)
                a_rep+=1
            
            if a_rep >=max_rep:
                return best_dist,best_path
            

            for j in range(k):
                start = 0

                if random_loc == True:
                    start = np.random.randint(len(A))
                
                dist_temp,path_temp,C_temp,valid = self.ant_path(A,P,Q,start)

                if update =='all' and valid == True:
                    C_all+= C_temp
                elif update =='elite' and valid == True:
                    elite_dist,elite_C = self.update_Elite(elite_dist,elite_C,ratio,k,dist_temp,C_temp)
                
                #getting best value per iteration#
                if dist_temp < iter_dist:
                    iter_dist = dist_temp
                    iter_path = path_temp
                    iter_C = C_temp
            
            

            #handling repeating values#
            if(np.abs(iter_dist-best_dist)<=tol):
                rep+=1
            else:
                rep = 0

            #updating the best iteration#
            if iter_dist<=best_dist:
                best_dist = iter_dist
                best_path = iter_path
                best_C = iter_C
            

            if update == 'best':
                X = self.updatePheromone(p,X,best_C,i,n)
                #X = (1-p)*X + best_C
            elif update == 'all':
                X = self.updatePheromone(p,X,C_all,i,n)
                #X = (1-p)*X + C_all
            elif update == 'elite':
                C_new = self.get_elite_C(elite_C)
                X = self.updatePheromone(p,X,C_new,i,n)
                
            

            if log == True:
                print("================================== \nbest distance at iteration {} : {} \n current best :{}".format(i,iter_dist,best_dist))
            

            if rep == max_rep:
                break
        

        #reorder path#
        best_path = self.reorder_path(best_path,start=0)
        return best_dist,best_path
###########################################################################################################################
    def get_distance_matrix_symmetric(self,path):

        """
            Creates a distance matrix based on the data proved in path

            Inputs:

                path : the path to the text file containing information on each node
            
            Outputs:
                A    : the distance matrix
        """
        
        with open(path) as reader :

            first_lines = 0
            i = 0
            X = []

            for lines in reader.readlines():

                if(first_lines>5 and lines!='EOF' and lines!='EOF\n'):
                    
                    stripped_line = lines.strip()
                    list_line = stripped_line.split()
                    w = [float(i) for i in list_line[1:]]
                    X.append(w)

                first_lines+=1
        
        X = np.array(X)
        m,n = X.shape

        A = []

        for i in range(0,m,1):
            u = X[i,:]
            dist = []

            for j in range(0,m,1):

                if i!=j:
                    v = X[j,:]
                    d = distance.euclidean(u,v)
                    dist.append(d)
                else:
                    dist.append(np.inf)

            A.append(dist)
        
        A = np.array(A)
        return A
####################################################################################################################
    def new_random(self,size,beta_max,alpha_max,max_iter,Q_max):
        """
            generates new random configuration
        """
        
        c = []

        #evaporation#
        p = np.random.uniform(0,1)
        c.append(p)

        #alpha#
        alpha = np.random.uniform(0,alpha_max)
        c.append(alpha)

        #beta#
        beta = np.random.uniform(0,beta_max)
        c.append(beta)

        #iterations#
        n = round(np.random.uniform(1,max_iter))
        c.append(n)

        #agents#
        k = round(np.random.uniform(1,size))
        c.append(k)

        #Q value#
        Q = np.random.uniform(0,Q_max)
        c.append(Q)

        return c
############################################################################################################################## 
    def get_cost(self,A,c,rep,size,beta_max,alpha_max,max_iter,Q_max):

        current_cost = 0

        for i in range(rep):
            temp_cost,_ = self.ACO(A,p=c[0],alpha=c[1],beta=c[2],n=c[3],k=c[4],Q=c[5],random_loc=True,update="best",ratio=0,max_rep=20,tol=0.001,log=False)
            current_cost+=temp_cost
        
        current_cost /=rep

        return current_cost
###########################################################################################################################################
    def SA(self,A,c0,rep,Tmax,Tmin,epoch,size,beta_max,alpha_max,max_iter,Q_max,log):
        """
            finds the global optimal configuration using the SA algorithm

            Inputs:
                A         : the distance matrix
                c0        : initial Configuration of the probelm
                Tmax      : maximum temperature
                Tmin      : minimum temperature
                epoch     : number of repetitions
                size      : number of nodes
                beta_max  : maximum allowed beta value
                alpha_max : maximum allowed alpha value
                max_iter  : maximum allowed iterations
                Q_max     : 
        """

        c = c0
        current_cost = self.get_cost(A, c, rep, size, beta_max, alpha_max, max_iter, Q_max)

        for T in range(Tmax,Tmin,-1):

            for i in range(0,epoch,1):

                #current_cost = get_cost(A, c, rep, size, beta_max, alpha_max, max_iter, Q_max)
                c_new = self.new_random(size,beta_max,alpha_max,max_iter,Q_max)
                new_cost = self.get_cost(A, c_new, rep, size, beta_max, alpha_max, max_iter, Q_max)
                delta_c = (-1*new_cost) - (-1*current_cost)

                p = np.random.uniform(0,1)

                if delta_c>0:
                    c = c_new
                    current_cost=new_cost
                elif (np.exp(delta_c/T)>p):
                    c = c_new
                    current_cost=new_cost
        
            if log==True:
                print("Temperature: {}".format(T),"\n=================================\n","current config:\n",c,"\n cost: {}\n".format(current_cost),"\n=================================\n")

        return c
###################################################################################################################################################        