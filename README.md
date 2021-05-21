# Adaline_Model
 Implements Adaline Model

Usage:
## Create Adaline:
adaline = makeAdaline(eta, n, fn, trace)
where eta is learning rate, n is number of intended inputs of the adaline, fn is initialization thunk and trace is trace value.

## Train Adaline:
train(adaline,dataset,epochs)

## Apply on single sample input:
applyAdalineVec(adaline,inputVector)
