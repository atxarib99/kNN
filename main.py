'''
    The main file. Sets up a kNN and tests its accuracy. Provided as an example.
'''

#get trainer and model
import kNN as knn
import TitanicParser
#using pyplotlib to plot error with k
import matplotlib.pyplot as plt

#load trainer from knn
trainer = knn.knn()

#get train and test data
X,Y,testX,testY,validX,validY = TitanicParser.loadData(validationSet=True)

#load train data
trainer.loadData(X,Y)

#holds error on validation set for each k
validErrors = {}
#setup k's to test
ks = range(1, 25)
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