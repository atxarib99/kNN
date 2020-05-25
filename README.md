# kNN

## Goals

This repository setups up a basic dry kNN (k-Nearest Neighbor). The idea is to have a structure where training data can quickly be loaded into predictions can be made.

## Example

The most basic example is provided below

```python
#import first
import knn

#setup trainer, k can be defined here
trainer = knn.knn(k=5)

#load training data
trainer.load(X,Y)

#make a prediction
guess = trainer.predict(testX)
```

A more complicated example, tuning the hyperparameter k is provided in main.py

## Documentation

All documentation has been provided in docstrings. Any merge requests should have docstrings included.
