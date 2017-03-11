# -*- coding : utf-8 -*- 
import numpy as np
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import tuple_dataset

# additional import
import sys

def readcsv(filename):
    data = np.loadtxt(filename, delimiter=',')
    # if there's a header, skip it
    #data = np.loadtxt(filename,delimiter=',', skiprows=1)
    return data 

class MnistModel(chainer.Chain):
    def __init__(self,n_units, n_out):
        super(MnistModel,self).__init__(
#   784 -> 100
#   100 -> 100
#   100 -> 10
            l1 = L.Linear(None,n_units),
            l2 = L.Linear(None,n_units),
            l3 = L.Linear(None,n_out),
        )

    def __call__(self,x):
        h = F.relu(self.l1(x)) 
        h = F.relu(self.l2(h)) 
        return self.l3(h)

if __name__ == '__main__':

# Data Reading
    Y_NUM = 10
    filename = sys.argv[1]
    csvdata = readcsv(filename)
    X = csvdata[:,1:]
    Y = csvdata[:,0]

# Data Converting
    X = X.astype(np.float32)/255
    Y = Y.astype(np.int32)
    N = Y.size
    x_train, x_test = np.split(X,[int(N*4/5)])
    y_train, y_test = np.split(Y,[int(N*4/5)])
    train = tuple_dataset.TupleDataset(x_train, y_train)
    test = tuple_dataset.TupleDataset(x_test, y_test)

# Initializing model
    model = L.Classifier(MnistModel(784,10))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

# Setting Iteration
    train_iter = chainer.iterators.SerialIterator(train,100)
    test_iter = chainer.iterators.SerialIterator(test,100,repeat=False,shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = training.Trainer(updater, (100,'epoch'), out="result")
    trainer.extend(extensions.Evaluator(test_iter, model, device=-1))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch','main/loss','validation/main/loss','main/accuracy','validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()
    serializers.save_npz('3perceptron.model',model) 
    print('end')
