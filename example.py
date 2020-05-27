'''
    This file serves to be an example on how to use kNN. This file is more complicated due to graphing. For a simpler example with just prediction, check examplesimple.py
    Change k, noise, and amount of data to see how accuracy is affected.
'''

import kNN as knn
import modeler
#using pyplotlib to plot error with k
import matplotlib.pyplot as plt
#needed for legend
import matplotlib.lines as mlines

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
#holds guesses where we guessed a 1
guessOneX = []
guessOneY = []
#holds guesses where we guessed a 2
guessZeroX = []
guessZeroY = []

#for each test element 
for i in range(len(testParameters)):
    #use trainer to get a guess
    confidence,guess = trainer.predict(testParameters[i], negativeValue=0)
    #cast guess to 1 or 0. Add the test element to corresponding list
    if guess == 0:
        guessZeroX.append(testParameters[i][0])
        guessZeroY.append(testParameters[i][1])
    else:
        guessOneX.append(testParameters[i][0])
        guessOneY.append(testParameters[i][1])
    #check if we were incorrect
    if guess != testLabel[i]:
        error += 1

#calcuate and print error
print("Accuracy", 1 - error / len(testParameters))

#create list of X,Y where labels are 1
oneX = []
oneY = []
for i in range(len(parameters)):
    if labels[i] == 1:
        oneX.append(parameters[i][0])
        oneY.append(parameters[i][1])
#create list of X,Y where lables are 0
zeroX = []
zeroY = []
for i in range(len(parameters)):
    if labels[i] == 0:
        zeroX.append(parameters[i][0])
        zeroY.append(parameters[i][1])
  
#get true function
trueX, trueY = modeler.trueFunction()

#plot truefunction
plt.plot(trueX, trueY, 'k+')
#plot training data with label 1
plt.plot(oneX, oneY, 'b+')
#plot training data with label 0
plt.plot(zeroX, zeroY, 'r+')
#plot test data with guess 1
plt.plot(guessOneX, guessOneY, 'bo')
#plot test data with guess 0
plt.plot(guessZeroX, guessZeroY, 'ro')

#legends
kplus = mlines.Line2D([], [], color='black', marker='+', markersize=15, label='True Function')
oneTrain = mlines.Line2D([], [], color='blue', marker='+', markersize=15, label='Positive Training Data')
zeroTrain = mlines.Line2D([], [], color='red', marker='+', markersize=15, label='Negative Training Data')
oneGuess = mlines.Line2D([], [], color='blue', marker='o', markersize=15, label='Positive Guess')
zeroGuess = mlines.Line2D([], [], color='red', marker='o', markersize=15, label='Negative Guess')

#set scale of graph to [0,1000]. This helps visualization of guesses
plt.xlim(0,1000)
plt.ylim(0,1000)

#display legend, title and show
plt.title("Accuracy: " + str(1 - error / len(testParameters)))
plt.legend(handles=[kplus, oneTrain, zeroTrain, oneGuess, zeroGuess], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)
plt.show()