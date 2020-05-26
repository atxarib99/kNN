'''
    This class will serve to generate a set of data to be used for kNN.
    The true function is y = .00001(X-500)^3 - .00015(X-500)^2 + 500. Why? idk, it looked cool.
'''

import random

def generateData(numElements, noiseFactor=.05):
    '''
        Generates data based on the true function. Adds noise depending on provided parmameter

        Parameters:
            numElements: The number of elements to generate
            noiseFactor: Percentage of mislabeled elements
        
        Result:
            A list of elements that contain tuples in order X,Y,label
    '''

    #generate numElements random elements
    elements = []
    x = []
    labels = []
    for i in range(numElements):
        #generate random X [0,1000] and Y [0,1000]
        X = random.randint(176, 925)
        Y = random.randint(176, 925)
        #get label from true function
        label = getLabel(X,Y)
        #get flipper to introduce noise
        flipper = random.random()
        #if the flipper meets the noise threshold
        if flipper < noiseFactor:
            #flip label
            if label == 0:
                label = 1
            else:
                label = 0
        #find label
        #add as tuple elements
        x.append(list((X,Y)))
        labels.append(label)
    return x,labels

def getLabel(X, Y):
    value = .00001 * ((X - 500) ** 3) - .0015 * ((X-500) ** 2) + 500
    if Y > value:
        return 1
    else:
        return 0

def trueFunction():
    '''
        Returns the true function as a set of points

        Parameters:
            None

        Result:
            X: the domain values
            Y: the range value
    '''

    x = []
    y = []
    for X in range(176, 925):
        value = .00001 * ((X - 500) ** 3) - .0015 * ((X-500) ** 2) + 500
        x.append(X)
        y.append(int(value))
    return x,y
