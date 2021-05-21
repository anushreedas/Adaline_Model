"""
adaline.py

This program implements an adaline model

@author: Anushree Das (ad1707)
"""
from utils import *

Adaline = namedtuple('Adaline',['eta', 'weightColMat', 'trace'])


def makeAdaline(eta,n,fn,trace):
    """
    Returns a Adaline named tuple with the given parameters

    :param eta: learning rate
    :param n:   number of intended inputs of the adaline
    :param fn:  initialization thunk
    :param trace: trace value
    :return:
    """
    return Adaline(eta=eta, weightColMat=makeMatrix(n+1,1,fn), trace=trace)


def sigma(x):
    """
    Implements the step function for adaline.
    It returns 1 if value is greater than 0,
    returns -1 if value is less than 0 and
    returns 0 if value is equal to 0
    :param x:   value
    :return:    1 or 0 or -1
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def applyAdaline(adaline,augColMatrix):
    """
    Returns the output of the adaline
    :param adaline:  adaline model
    :param augColMatrix:list of input vectors
    :return:            output of the adaline
    """
    for element in augColMatrix.data:
        if element != -1 and element != 1:
            raise TypeError('The augmented column matrix should contain only 1s and -1s')
    # returns sigma(w.V) where w is weights of the model and V is a list of input vectors
    return sigma(dot(adaline.weightColMat,augColMatrix))


def applyAdalineVec(adaline,inputVector):
    """
    Returns the output of the adaline
    :param adaline:     adaline model
    :param augColMatrix:input vector
    :return:            output of the adaline
    """
    for element in inputVector.data:
        if element != -1 and element != 1:
            raise TypeError('The input vector should contain only 1s and -1s')
    # convert input vector to column matrix and augment the column matrix
    augColMatrix = augmentColMat(colMatrixFromVector(inputVector))
    # returns sigma(w.v) where w is weights of the model and n is a input vector
    return sigma(dot(adaline.weightColMat,augColMatrix))


def trainOnce(adaline,inputVector,targetOutput):
    """
    Applies the adaline learning rule to the adaline model
    :param adaline:  adaline model
    :param inputVector: sample input vector
    :param targetOutput:target output
    :return: None
    """
    if adaline.trace == 2:
        print('On sample: input=',inputVector,'target=',targetOutput,',',adaline)

    # convert input vector to column matrix and augment the column matrix
    augColMatrix = augmentColMat(colMatrixFromVector(inputVector))
    # find w.V where w is weights of the model and V is a list of input vectors
    y0 = dot(adaline.weightColMat,augColMatrix)
    # get difference between y0 and the actual output
    delta = targetOutput - y0
    # find delta of weights
    deltaWeights = adaline.eta * delta * augColMatrix
    # update weights
    setMat(adaline.weightColMat,add(adaline.weightColMat, deltaWeights))


def andDataSetCreator():
    """
    Creates AND dataset for adaline model
    :return: AND dataset
    """
    data = []
    for i in range(1,-2,-2):
        for j in range(1,-2,-2):
            data.append((Vector(data=[i,j]),i | j))
    return data


andDataSet = andDataSetCreator()


def orDataSetCreator():
    """
    Creates OR dataset for adaline model
    :return: OR dataset
    """
    data = []
    for i in range(1,-2,-2):
        for j in range(1,-2,-2):
            data.append((Vector(data=[i,j]),i & j))
    return data


orDataSet = orDataSetCreator()


def trainEpoch(adaline,dataset):
    """
    Trains the adaline once for each entry in the data set
    :param adaline:  adaline model
    :param dataset:  data set on which to train on
    :return:         None
    """
    # train for each row in dataset
    for sample in dataset:
        trainOnce(adaline, sample[0], sample[1])

    if adaline.trace == 1:
        print('After epoch: ', adaline)


def train(adaline,dataset,epochs):
    """
    Trains adaline on given dataset iteratively and
    terminates when the number of epochs exceeds the bound.
    :param adaline:  adaline model
    :param dataset:  data set on which to train on
    :param epochs:   number of training epochs
    :return:         None
    """
    iteration = 0
    # terminates when the number of epochs exceeds the bound
    while iteration <= epochs:
        trainEpoch(adaline,dataset)
        iteration += 1
