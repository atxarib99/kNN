'''
    This file holds the class which represents a training example and the definition of euclidean distance
'''

#math is necessary for this class
import math

def eucdist(X1, X2):
    '''
        Euclidean Distance. sqrt( (x1,0 - x2,0)^2 + ... + (x1,n - x2,n)^2 ), where n is number of parameters/dimensions.

        Parameters:
            X1: The parameters in the first object we are comparing as a list
            X2: The parameters in the second object we are comparing as a list
        
        Result:
            The euclidean distance between the two points
    '''
    #holds the distance
    dist = 0
    #for each parameter/dimension in each item
    for i in range(len(X1)):
        #add the square of the difference between the i-th parameter in each of the items we are comparing
        dist += (X2[i] - X1[i]) ** 2
    
    #return the squareroot of the dist so far
    return math.sqrt(dist)

class item:
    '''
        This class holds an item in the dataset, both parameters and label.
    '''

    def __init__(self, X,Y=-1,distFunc=eucdist):
        '''
            The constructor for the item class. Requires parameters of a single training example. 
            A label should be provided if it is a true training example.
            A distance function can also be provided. If not, euclidean distance is used.

            Parameters:
                X: Parameters of training example
                Y: Label of training example, default -1
                distFunc: Distance function to be used for this instance, default euclidean distance.
            
            Result:
                Instance of item class
        '''
        super().__init__()
        self.distFunc = distFunc
        self.X = X
        self.label = Y
    
    def dist(self, other):
        '''
            Runs the provided distance function for this class, comparing a provided item.

            Parameters:
                Other: The other item to compare to
             
            Result:
                The distance between the instance of this class and the instance of an item class provided as an parameter.
        '''
        return self.distFunc(self.X, other.X)


    