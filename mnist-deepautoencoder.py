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

lmj.cli.enable_default_logging()

URL = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
DATASET = os.path.join(tempfile.gettempdir(), 'mnist.pkl.gz')

if not os.path.isfile(DATASET):
    logging.info('downloading mnist digit dataset from %s' % URL)
    urllib.urlretrieve(URL, DATASET)
    logging.info('saved mnist digits to %s' % DATASET)

train, valid, _ = [x for x, _ in cPickle.load(gzip.open(DATASET))]

e = theanets.Experiment(theanets.Autoencoder,
                        #layers=(784, 250, 784), learning_rate=.01, learning_rate_decay=.2, patience=10, optimize="sgd",
                        #layers=(784, 250, 150, 250, 784), learning_rate=.01, learning_rate_decay=.2, patience=10, optimize="sgd",
                        layers=(784, 250, 150, 30, 150, 250, 784), learning_rate=.005, learning_rate_decay=.1, patience=20, optimize="sgd",
                        #Going deeper gave worse performance, despite having a larger code layer, for these hyperparameters
                        #layers=(784, 250, 150, 100, 60, 100, 150, 250, 784), learning_rate=.001, learning_rate_decay=.05, patience=40, optimize="sgd",
                        num_updates=256,
                        tied_weights=True,
                        batch_size=32,
                        )
e.run(train, valid)
e.save("deepauto.pkl")

pred = e.network(valid)

for i in np.random.randint(pred.shape[0], size=5):
    dim = int(np.sqrt(pred.shape[1]))
    fig, axarr = plt.subplots(1,2)
    axarr[0].set_title("Original")
    axarr[0].imshow(valid[i,:].reshape((dim,dim)), cmap="gray")
    axarr[1].set_title("Reconstructed")
    axarr[1].imshow(pred[i,:].reshape((dim,dim)), cmap="gray")
plt.show()
