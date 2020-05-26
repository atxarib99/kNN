'''
    This example file shows how to tune the hyper parameter k.
'''

#get trainer and model
import kNN as knn
import modeler as model
#using pyplotlib to plot error with k
import matplotlib.pyplot as plt

#get trainer
trainer = knn.knn()

#get training data, with some noise
X,Y = model.generateData(100, noiseFactor=.2)

#get validation data, we will assert that this data has no noise, even though this is not accurate in real data
validX, validY = model.generateData(25, noiseFactor=0)

#get test data, also with no noise
testX, testY = model.generateData(25, noiseFactor=0)

#load training data
trainer.loadData(X,Y)

#holds error on validation set for each k
validErrors = {}
#setup k's to test
ks = range(1, 100, 2)
for k in ks:
    #holds the error count for this k
    error = 0
    #set the k parameter in the trainer
    trainer.setK(k)
    #for each validation example
    for i in range(len(validX)):
        #try to predict its label using training data
        guess = trainer.predict(validX[i])
        #cast guesses to T/F
        if guess > .5:
            guessRound = 1
        else:
            guessRound = 0
        #check if wrong
        if guessRound != validY[i]:
            error += 1
    #save error
    validErrors[k] = (error / len(validX))
    print("Validation Error", k, (error / len(validX)))

#find best k
error = 1
bestK = 1
#for each k tried
for key in validErrors.keys():
    #if this k is better, save it
    if validErrors[key] < error:
        error = validErrors[key]
        bestK = key

#set k to trainer
trainer.setK(bestK)


#holds errors
error = 0
#for each item in test data
for i in range(len(testX)):
    #try to predict its label using training data
    guess = trainer.predict(testX[i])
    #cast guesses to T/F
    if guess > .5:
        guessRound = 1
    else:
        guessRound = 0
    #check if wrong
    if guessRound != testY[i]:
        error += 1

#print error and accuract
print("K:", bestK)
print("Error:", error / len(testX))
print("Accuracy:", 1 - error / len(testX))

#holds errors related to each k
errors = []
#for each k, add its respective error to errors list
for k in ks:
    errors.append(validErrors[k])
#plot, k and errors
plt.plot(ks, errors)
plt.show()