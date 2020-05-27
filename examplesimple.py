'''
    This file serves to be an example on how to use kNN. This file is the simplified version with no graphs. For an example with graphs, check example.py
    Change k, noise, and amount of data to see how accuracy is affected.
'''

import kNN as knn
import modeler

#setup the trainer. Tune your k parameter here.
trainer = knn.knn(k=5)

#using basic modeler provided, can define how many elements, and how much noise we want.
parameters, labels = modeler.generateData(100, noiseFactor=.25)

#get parameters to test on. These should have 0 noise so we can accurately test them
testParameters, testLabel = modeler.generateData(25, noiseFactor=0)

#load the train data into the trainer
trainer.loadData(parameters, labels)

#holds the number of incorrect
error = 0
#for each test element 
for i in range(len(testParameters)):
    #use trainer to get a guess
    confidence,guess = trainer.predict(testParameters[i], negativeValue=0)
    #check if we were incorrect
    if guess != testLabel[i]:
        error += 1

#calcuate and print error
print("Accuracy", 1 - error / len(testParameters))