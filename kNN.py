'''
    This file serves the purpose of the main computation required for kNN (Nearest Neighbor)
'''
#we will need to import sys and the item class
import sys
import item

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
            trainingExamples.append(item.item(X[i], Y[i]))
        
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

    
    def predict(self, X):
        '''
            Given parameters, makes a prediction using training data

            Parameters:
                X: Parameters for prediction
            Result:
                Y: returns a prediction using kNN.
        '''
        #convert X into an item object
        toPredict = item.item(X, -1)
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
        #return average
        return avg / self.nn