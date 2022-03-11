#Imports
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#np.seterr(all='raise')
import pandas as pd
import tsplib95
import networkx as nx
import matplotlib.pyplot as plt

class Ant:
    def __init__(self):
        print("loaded")

############### Ant Colony Optimistion Algorithm ############################################
    def nearest_neighbour(self,D,Q):
        """
            determines the shortest path using the nearest neighbour heuristic.
            will be used as an initial configuration for ACO

            Inputs:
                D : the distance matrix
                Q : pheromone update weight
            
            Outputs:
                path : the suboptimal path to be taken
        """
        A = D.copy()
        n = len(A)

        dist = 0
        curr_pos = 0
        next_pos = 0

        path = [curr_pos]

        count = n

        C = np.ones((n,n))

        while count>0:

            b = np.copy(A[curr_pos,:])

            next_pos = np.argmin(b)
            path.append(next_pos)
            dist+= A[curr_pos,next_pos]

            for i in range(1,len(path)-1,1):
                A[next_pos,path[i]] = np.inf
            
            if count>2:
                A[next_pos,0] = np.inf

            curr_pos = next_pos
            count-=1
        

        #pheromone trail#

        for i in range(0,len(path)-1,1):
            j = path[i]
            k = path[i+1]
            C[j,k] += Q/dist

        
        return dist,path,C
    
    def get_prob_matrix(self,Distance,Pheromone,alpha,beta):
        """
            Constructs the probability matrix depending on the
            distance matrix and pheromone matrix

            Inputs:
                Distance  : the distance vector 
                Pheromone : the pheromone vector
                alpha     : the parameter over the pheromone
                beta      : the parameter over
            Outputs:
                C         : the probability vector
        """

        A = Distance.copy()
        P = Pheromone.copy()

        n = len(A)
        C = np.zeros(n)

        for i in range(0,n,1):
            C[i] = (P[i]**alpha) * ((1/A[i])**beta)
        
        deno = 1/(sum(C))
        C = deno*C

        return C
    
    def accumulator(self,prob_vector):
        """
            returns index of range where random value lies in

            Inputs:
                prob_vector : the probability of going to next state
        """

        v = prob_vector.copy()
        n = len(v)
        u = np.zeros(n)
        index = -1
        r = np.random.uniform(0,1,1)[0]

        for i in range(0,n,1):
            u[i] = sum(v[i:])

        for i in range(0,n,1):

            if (i<n-1) and (r>u[i+1]) and (r<=u[i]):
                index = i
                break
            elif (i==n-1) and (r>0) and (r<=u[n-1]):
                index = i
                break
        
        return index

    def ant_path(self,Distance,Pheromone,alpha,beta,Q,start):
        """
            path that an agent will depending on the input
            parameters

            Inputs:
                Distance  : the Distance Matrix
                Pheromone : the Pheromone Matrix
                alpha     : the parameter over the pheromone
                beta      : the parameter over
                Q         : the pheromone update weight
                start     : the start index of the agent

            Outputs:
                dist      : distance traveled by agent
                path      : path taken by agent
                C         : pheromone agent deposited
                valid     : if solution is a valid solution or not
        """

        A = Distance.copy()
        P = Pheromone.copy()

        n = len(A)
        i = start
        count = n
        valid = True

        dist = 0
        path = [i]
        C = np.zeros((n,n))

        while (count>0) and (valid == True):

            #getting probability vector#
            v = self.get_prob_matrix(A[i,:],P[i,:],alpha,beta)
            j = self.accumulator(v)

            if j == -1:
                valid = False
                break

            path.append(j)
            dist+= A[i,j]

            for k in range(1,len(path)-1,1):
                A[j,path[k]] = np.inf
            
            if count>2:
                A[j,start] = np.inf

            i = j
            count-=1
        
        #updating pheromone#

        if valid == True:
            for m in range(0,len(path)-1,1):
                p = path[m]
                k = path[m+1]
                C[p,k] = Q/dist
        
        return dist,path,C,valid
    
    def reorder_path(self,path,start):
        """
            reorders path starting and ending with start index

            Input:
                path     : path which is a permuatation that will be reordered
                start    : the starting index 

            outputs:
                new_path : reordered path
        """
        if not(start in path):
            return path
        
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
    

    def update_C(self,C_best,C_list,dist_list,update,n,num_agent,ratio):
        """
            calculates update pheromone matrix
        """

        C = C_list.copy() 
        dist = dist_list.copy()

        if update == 'best':
            C_ans = np.zeros((n,n))

            for i in range(0,len(C_best),1):
                C_ans = C_ans + C_best[i]
            
            return C_ans

        elif update == 'all':

            C_ans = np.zeros((n,n))

            for i in range(0,len(C),1):
                C_ans = C_ans + C[i]
            
            return C_ans
        else:

            #sort the C_list#
            m = len(C)
            for i in range(0,m-1,1):
                for j in range(i+1,m,1):

                    if dist[j] < dist[i]:
                        temp_d = dist[i]
                        dist[i] = dist[j]
                        dist[j] = temp_d

                        temp_c = C[i]
                        C[i] = C[j]
                        C[j] = temp_c
            
            num = int(np.round(len(C)*ratio))
            C_ans = np.zeros((n,n))

            for i in range(0,len(C),1):

                if i < num:
                    C_ans = C_ans + 2*C[i]
                else:
                    C_ans = C_ans + C[i]
            

            
            return C_ans


    def updatePheromone(self,p,Pheromone,update_C,i,n):
        """
            updates the pheromone depending on parameters
        """

        s = p
        if p == -1:
            s = 1 - (i+1)/n
        ans = (1-s)*Pheromone + update_C

        return ans
    
    def global_elite(self,gelite,distelite,temp_dist,temp_C,n_ants,ratio):

        m = int(round(n_ants*ratio))
        cElite = gelite.copy()
        distElite = distelite.copy()
        C = temp_C.copy()
        dist = temp_dist.copy()

        if len(cElite)==0:
            cElite.append(C)
            distElite.append(dist)
            return cElite,distElite

        else:
            k = len(cElite) - 1

            while k>=0:

                if dist<distElite[k]:
                    k-=1
                else:
                    break
            
            cElite.insert(k+1,C)
            distElite.insert(k+1,dist)

            if len(cElite)>m:
                cElite.pop(-1)
                distElite.pop(-1)

            return cElite,distElite


    def ACO(self,Distance,p,alpha,beta,n,k,Q,random_loc,update,ratio,max_rep,tol,log,plot,opt):
        """
            Finds optimal route using Ant Colony Optimisation techniques
            
            Inputs:
                Distance    : Distance Matrix
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
        
        A = Distance.copy()
        A = A.astype('float64')
        rep = 0
        min_dist,min_path,P = self.nearest_neighbour(A, Q)
        min_C = P
        plot_error1 = []
        plot_error2 = []
        max_iter = 0
        cElite = []
        distElite = []

        #looping through each generation#
        for i in range(0,n,1):
            max_iter+=1
            dist_list = []
            C_list = []

            iter_dist = None
            iter_path = None
            iter_C = None


            #looping through agents#
            for j in range(0,k,1):

                if random_loc == True:
                    start = np.random.randint(len(A))
                
                dist_temp,path_temp,C_temp,valid = self.ant_path(A, P, alpha, beta, Q, start)


                #adds to list if solution is valid#
                if valid == True:

                    #update global values#
                    if update =='best':
                        cElite,distElite = self.global_elite(cElite,distElite,dist_temp,C_temp,k,ratio)
                    
                    #updates best solution in current generation#
                    if (len(dist_list) == 0) or (dist_temp < iter_dist):
                        iter_dist = dist_temp
                        iter_path = self.reorder_path(path_temp, start=0)
                        iter_C = C_temp

                    dist_list.append(dist_temp)
                    C_list.append(C_temp)
                        
            
            #updates if iteration best is better than global best#
            if (iter_dist!=None) and (iter_dist<min_dist):
                min_dist = iter_dist
                min_path = iter_path
                min_C = iter_C
        
            #checking if solution repeats multiple times#
            if (iter_dist!=None) and (np.abs(iter_dist-min_dist)<=tol):
                rep+=1
            else:
                rep = 0
            
            if log == True:
                print("================================== \nbest distance at iteration {} : {} \n current best :{}".format(i,iter_dist,min_dist))


            #plots solution#
            if plot == True and iter_dist!=None and min_dist!=None:
                plot_error1.append((np.abs(min_dist-opt)/opt)*100)
                plot_error2.append((np.abs(iter_dist-opt)/opt)*100)

            #exit if solution the same#
            if rep == max_rep:
                break
            
            #getting C matrix based on update rule#
            C_update = self.update_C(cElite, C_list, dist_list, update, len(A),k, ratio)

            #update pheromone matrix#
            P = self.updatePheromone(p, P,C_update, i, n)

        if plot == True:

            #saving values to textfile#
            outfile = open("error.txt","w")
            outfile.write('[')
            for i in range(0,len(plot_error1)-1,1):
                outfile.write(str(plot_error1[i]) + ",")
            
            outfile.write(str(plot_error1[-1])+']')
            outfile.close()

            plt.rcParams['figure.figsize'] = (16,10)
            x_axis = list(range(max_iter))
            plt.plot(x_axis,plot_error1,label = r'error % of best solution')
            plt.plot(x_axis,plot_error2,label= r"error % at each iteration")
            #plt.ylim(0,100)
            plt.xlabel('iteration')
            plt.ylabel(r'error %')
            plt.legend(loc='best')
            plt.savefig('output.png')
            plt.close()
        return min_dist,min_path
    
    def get_distance_matrix(self,path):
        """
            gets the distance matrix using the tsplib
        """

        probelm = tsplib95.load(path)
        graph = probelm.get_graph()
        A = nx.to_numpy_matrix(graph)
        A = np.squeeze(np.asarray(A))

        n = len(A)

        for i in range(0,n,1):
            A[i,i] = np.inf
        
        return A
    
    def get_graph(self,file_link,path,A):
        """
            shows the path taken visually
        """

        size = len(A)
        probelm = tsplib95.load(file_link)
        graph = probelm.get_graph()
        graph.remove_node(size)
        edges = graph.edges()
        #graph.remove_edges_from(edges)


        for i in range(0,len(path)-1,1):
            graph.add_edge(path[i], path[i+1],weight=A[path[i],path[i+1]])

        nx.draw_networkx(graph,edge_color ='red')
        plt.savefig("graph.png")
############### Ant Colony Optimistion Algorithm ############################################

##################Simulated Annealing Algorithm #############################################
    def new_config(self,x_state,dim,scale):
        """
            generates new state for simulated annealing

            Inputs:
                x_state : the current the state 
                dim     : the dimension of the distance matrix
                scale   : scaling parameter

            Output:
                y       : the new state
        """

        x = x_state.copy()
        y = []

        #getting p#
        p = x[0] + np.random.uniform(-0.5,0.5,size=1)[0]

        if p>1:
            p = 0.5
        elif p<0:
            p = 0.5
        
        y.append(p)

        #getting alpha#
        alpha = x[1] + np.random.uniform(-1,1,size=1)[0]

        if alpha < 0:
            alpha = 0
        
        y.append(alpha)

        #getting beta#
        beta = x[2] + np.random.uniform(-1,1,size=1)[0]

        if beta < 0:
            beta = 0
        
        y.append(beta)

        #getting n#
        #n = x[3] + int(np.round(np.random.uniform(-1,1,size=1) * scale))
        n = x[3]
        if n<0:
            n = 0
        
        y.append(n)

        #getting k#
        k = x[4] + int(np.round(np.random.uniform(-1,1,size=1) * np.sqrt(dim)))

        if k<0:
            k = 0
        
        y.append(k)

        #getting Q#
        Q = x[5] + np.random.uniform(-1,1,size=1)[0]

        if Q < 1:
            Q = 1
        
        y.append(Q)
        return y
    
    def get_average(self,Distance,x0,update_type,ratio,tol,rep,loop):
        """
            runs aco multiple times and returns average cost

            Inputs:
                Distance    : the distance matrix
                x0          : current configuration of parameters
                update_type : how the pheromone matrix is updated (all,best,elite)
                ratio       : percentage of best solutions to be used (value between 0-1)
                tol         : tolerance for the ACO alpgrithm
                rep         : maximum number of repietitions aco will allow
                loop        : the amount of times algorithm will be run

        """
        ans = 0
        A = Distance.copy()
        x = x0.copy()

        for i in range(0,rep,1):
            fx,x_path = self.ACO(A,p=x[0],alpha=x[1],beta=x[2],n=x[3],k=x[4],Q=x[5],random_loc=True,update=update_type,ratio=ratio,max_rep=rep,tol=tol,log=False,plot=False,opt=0)
            ans+=fx
        
        ans = ans/rep

        return ans


    def SA(self,Distance,x0,t0,t1,n,alpha,update_type,ratio,tol,rep):
        """
            Determines suboptimal parameters for ACO on a dataset
            using simulated annealing

            Inputs:
                Distance    : the distance matrix
                x0          : intial configuration of parameters
                t0          : lower temperature bound small value
                t1          : upper temperature bound large value
                n           : number of iterations
                alpha       : temperature decrease rate (value between 0-1)
                update_type : how the pheromone matrix is updated (all,best,elite)
                ratio       : percentage of best solutions to be used (value between 0-1)
                tol         : tolerance for the ACO alpgrithm
                rep         : maximum number of repietitions aco will allow
        """

        A = Distance.copy()
        x = x0.copy()
        #fx,x_path = self.ACO(A,p=x[0],alpha=x[1],beta=x[2],n=x[3],k=x[4],Q=x[5],random_loc=True,update=update_type,ratio=ratio,max_rep=rep,tol=tol,log=False,plot=False,opt=0)
        fx = self.get_average(Distance, x0, update_type, ratio, tol, rep,loop = 5)

        while t1 > t0:
            scale = np.sqrt(t1-t0)
            
            for i in range(0,n,1):

                y = self.new_config(x,len(A),scale)
                #fy,y_path = self.ACO(A,p=y[0],alpha=y[1],beta=y[2],n=y[3],k=y[4],Q=y[5],random_loc=True,update=update_type,ratio=ratio,max_rep=rep,tol=tol,log=False,plot=False,opt=0)
                fy = self.get_average(Distance,y, update_type, ratio, tol, rep,loop=5)
                if fy < fx:
                    fx = fy
                    x = y.copy()
                elif np.random.rand() < np.exp(-(fy-fx)/t1):
                    fx = fy
                    x = y.copy()
            
            t1 = alpha * t1
        
        return x,fx

