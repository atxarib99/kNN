'''
    This file serves the purpose of the main computation required for kNN (Nearest Neighbor). It also holds the class which represents a training example and the definition of euclidean distance
'''

#math is necessary for this class
import math
#we will need to import sys
import sys

class knn:
    '''
        This class is an object that holds the training examples for kNN and how to make predictions from it.
    '''

    def __init__(self, k=5):
        '''
            Initializes a kNN class. k can be provided for how many training examples should be taken into account for predictions.
            Smaller k means that the predictions will be noisy.
            Too large of k means that items that are really far and uncorrelated are taking into account during prediction.

            Parameters:
                nn: how many nearest neighbors should be accounted for during predictions

            Result:
                Instance of the kNN class.
        '''
        super().__init__()

        self.nn = k

    def loadData(self,X,Y):
        '''
            Loads training data into the kNN class.
            TODO: make data appendable
            Parameters:
                X: a matrix of the parameters of the training examples
                Y: a matrix/list of the labels.
            Result:
                Adds data for the kNN to use for predictions. Returns None
        '''

        #holds training examples
        trainingExamples = []
        #for each example provided
        for i in range(len(X)):
            #add it as an item to the list
            trainingExamples.append(_item(X[i], Y[i]))
        
        #save the list of created examples in this instance
        self.examples = trainingExamples

    def setK(self, k):
        '''
            Changes the number of nearest neighbors to use for prediction.
            Note: It is advantageous to use a odd number to prevent ties.

            Parameters:
                k: the new k parameter
            Result:
                Updates the number of nearest neighbors to use. Returns None
        '''

        self.nn = k

    
    def predict(self, X, positiveValue = 1, negativeValue = -1):
        '''
            Given parameters, makes a prediction using training data

            Parameters:
                X: Parameters for prediction
                postiveValue: what should the value be if we guess True, usually 1.
                negativeValue: what should the value be if we guess False, usually 0 or -1.
            Result:
                Y: returns a prediction using kNN. This also provides a confidence amount, it does not match the labels.
                guess: returns a prediction that has been casted to match the provided labels as parameters.
        '''
        #convert X into an item object
        toPredict = _item(X, -1)
        #holds nearest neighbors
        neighbors = {}
        #build nearest neighbors
        for example in self.examples:
            #if neighbors isn't filled yet, fill it first
            if len(neighbors) < self.nn:
                dist = toPredict.dist(example)
                neighbors[dist] = example
            else:
                #get key set
                keySet = list(neighbors.keys())
                #find furthest neighbor
                greatestDist = 0
                for key in keySet:
                    if key > greatestDist:
                        greatestDist = key
                #if current example is closer than furthest, replace
                dist = toPredict.dist(example)
                if dist < greatestDist:
                    neighbors.pop(greatestDist)
                    neighbors[dist] = example
        #find average of labels
        keySet = neighbors.keys()
        avg = 0
        for key in keySet:
            avg += neighbors[key].label
        avg /= self.nn
        if avg < .5:
            guess = negativeValue
        else:
            guess = positiveValue
        #return average
        return avg, guess


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

class _item:
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

