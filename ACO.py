import numpy as np
import random

class ACO:
##########################################################################################################################################################    
    def updateProbMatrix(self,A,P,alpha,beta):
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
                
                if i!=j and A[i,j]!=0:
                    X[i,j] = (P[i,j]**alpha)*((1/A[i,j])**beta)/deno
                else:
                    X[i,j] = 0
        
        return X
########################################################################################################################################################
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
#########################################################################################################################################################
    def ant_path(self,A,Prob):
        """
            Calculates the path the ant with take

            Inputs:
                A    : the Distance matrix
                Prob : the probability matrix

            output:
                C    : a matrix with 1/d at the edges the ant passed. where d is the distance traveled 
        """

        n = len(A)
        P = np.copy(Prob)
        C = np.zeros((n,n))
        distance = 0
        path = []
        i = 0
        count = 0

        while count<n:

            v = P[i,:]
            
            #reducing the probability of going to start#
            if i!=0 and count<n-1:
                v[0] = np.min(v[np.nonzero(v)])/n

            j = self.accumulator(v)
            distance+= A[i][j]
            path.append([i,j])

            #making sure ant doesnt visit same node twice
            if i!=0:
                P[:,i] = 0

            i = j
            count+=1

        for x in range(0,len(path),1):
            [i,j] = path[x]
            C[i,j] += 1/distance   
        return C
############################################################################################################################################################    
    def final_walk(self,A,Prob):
        """
            Calculates the path the last ant with take

            Inputs:
                A    : the Distance matrix
                Prob : the probability matrix

            output:
                C    : a matrix with 1/d at the edges the ant passed. where d is the distance traveled 
        """

        n = len(A)
        P = np.copy(Prob)
        C = np.zeros((n,n))
        distance = 0
        path = []
        i = 0
        count = 0

        while count<n:

            v = P[i,:]
            
            #reducing the probability of going to start#
            if i!=0 and count<n-1:
                v[0] = np.min(v[np.nonzero(v)])/n

            j = self.accumulator(v)
            distance+= A[i][j]

            if j!=-1:
                path.append([i,j])

            if j == 0:
                break
            #making sure ant doesnt visit same node twice
            if i!=0:
                P[:,i] = 0

            # P[:,i] = 0
            i = j
            count+=1

        return distance,path
###########################################################################################################################################################
    def ACO(self,A,p,alpha,beta,n,k):
        """
        Finds optimal route using Ant Colony Optimisation techniques
        
        Inputs:
            A: Distance Matrix
            p: (scalar) evaporation rate
            alpha: (scalar) parameter that affects pheromone weighting
            beta: (scalar) parameter that affects distance weighting
            n: (scalar) number of interations to be performed
            k: (scalar) number of ants to be used
            
        Output:
            path: set of 2-tuples of route to be taken
        """

        #Pheromone matrix#
        X = np.ones((len(A),len(A)))

        for i in range(0,n,1):

            #pheromone update matrix#
            C = np.zeros((len(A),len(A)))

            #probability matrix#
            P = self.updateProbMatrix(A,X,alpha,beta)

            #constructing ant paths#
            for j in range(0,k,1):
                C+= self.ant_path(A,P)
            
            #updating pheromone#
            X = (1-p)*X + C
        
        dist,path = self.final_walk(A,P)

        while path[-1][1]!= 0 or len(path)!=len(A):
            dist,path = self.final_walk(A,P)

        return dist,path
################################################################################################################################################################
