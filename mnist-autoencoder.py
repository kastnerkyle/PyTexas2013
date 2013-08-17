#!/usr/bin/env python

import cPickle
import gzip
import logging
import lmj.cli
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import theanets
import urllib
from sklearn.linear_model import SGDClassifier

lmj.cli.enable_default_logging()

URL = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
DATASET = os.path.join(tempfile.gettempdir(), 'mnist.pkl.gz')

if not os.path.isfile(DATASET):
    logging.info('downloading mnist digit dataset from %s' % URL)
    urllib.urlretrieve(URL, DATASET)
    logging.info('saved mnist digits to %s' % DATASET)

train, valid, _ = [x for x, _ in cPickle.load(gzip.open(DATASET))]

e = theanets.Experiment(theanets.Autoencoder,
                        layers=(784, 150, 784),
                        tied_weights=True,
                        optimize="sgd",
                        )
e.run(train, valid)
e.save("auto.pkl")

encoded_train = e.network.forward(train)[-2]
encoded_valid = e.network.forward(valid)[-2]
train, valid, _ = [
    (x, y.astype('int32')) for x, y in cPickle.load(gzip.open(DATASET))]
clf = SGDClassifier()
clf.fit(*train)
print "Score on raw features:",clf.score(*valid)
clf.fit(encoded_train, train[-1])
print "Score on encoded features:",clf.score(encoded_valid, valid[-1])
