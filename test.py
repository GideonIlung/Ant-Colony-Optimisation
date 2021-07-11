#link to test data: http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/#
# link to results : http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/STSP.html#

#IMPORTS
import numpy as np
import unittest
from scipy.spatial import distance
from ACO import *



def get_distance_matrix(path):

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
                dist.append(0)

        A.append(dist)
    
    A = np.array(A)
    return A


def metric(path,opti_dist,epoch,p,alpha,beta,n,k):
    """
        creates a metric value that will be used to determine if algorithm is passing requirements or not

        Inputs:
            A         : the distance matrix
            opti_dist : the optimal distance 
            epoch     : number of times algorithm must be run
            p         : (scalar) evaporation rate
            alpha     : (scalar) parameter that affects pheromone weighting
            beta      : (scalar) parameter that affects distance weighting
            n         : (scalar) number of interations to be performed
            k         : (scalar) number of ants to be used

        Output:
            value     : value between 0-1 
    """

    A = get_distance_matrix(path)

    model = ACO()

    min_dist,path = model.ACO(A,p,alpha,beta,n,k)

    for i in range(1,epoch,1):

        dist,path = model.ACO(A,p,alpha,beta,n,k)

        if dist < min_dist:
            min_dist = dist
        
        
    value = (min_dist-opti_dist)/min_dist

    return value,min_dist


class Tests(unittest.TestCase):

    def test_0(self):
        path = "Data/st70/st70_tsp.txt"
        opti_dist = 675
        value,dist = metric(path,opti_dist,epoch=30,p=0.5,alpha=10,beta=5,n=40,k=8)

        print('\nTEST CASE 0')
        print('\noptimal distance    : ', opti_dist)
        print('calculated distance : ',dist)
        self.assertLessEqual(value,0.5,"Test 0 Failed")

    def test_1(self):
        path = "Data/eil101/eil101_tsp.txt"
        opti_dist = 629
        value,dist = metric(path,opti_dist,epoch=30,p=0.5,alpha=10,beta=5,n=40,k=8)

        print('\nTEST CASE 1')
        print('\noptimal distance    : ', opti_dist)
        print('calculated distance : ',dist)
        self.assertLessEqual(value,0.5,"Test 1 Failed")

    
    # def test_1(self):
    #     path = "Data/a280/a280_tsp.txt"
    #     opti_dist = 2579
    #     value = metric(path,opti_dist,epoch=5,p=0.5,alpha=1,beta=1,n=20,k=4)
    #     self.assertLessEqual(value,0.5,"Test 1 Failed")

if __name__ == '__main__':
    unittest.main()



        