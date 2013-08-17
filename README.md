PyTexas2013
===========

Code and slides for PyTexas 2013 talk, "Trends in Deep Learning"

To get the packages for this demo code, use the following commands:
pip install theano theanets scikit-learn 

There are some configuration details for running Theano on the GPU - see
http://deeplearning.net/software/theano/install.html . However, Theano will 
run on the CPU as well, so the GPU setup is purely optional. However, GPU code
is much, much, much, MUCH faster (though it does require an Nvidia card).


Interpreting log output
=======================

Theanets will output logs as it trains - interpreting this can be tricky.

For example, take the output line below
MainProcess theanets.trainer:76 SGD update 59/128 @1.33e-03 train [ 3.95852201  3.95852201  0.        ]

The line above shows the network is on update 59 out of 128 total sweeps through the data (called epochs).
The number after the @ sign shows the current value of the learning rate, which is set to decay over epochs by default
The output inside the [] depends on the type of network being trained.
The first value inside the [] is always the current value of the objective function -- 
the cost, plus any regularizers that are in effect. This value is what is being minimized.

The second value depends on the type of network you're training. If it's a Classifier, then the second value is the error, 
the fraction of examples classified incorrectly.

If it's not a classifier, then this second value is the raw cost function 
(usually the squared error), without any regularizers.

The remaining values indicate the sparsities of the hidden layers in the network 
-- the higher the number, the more of the units in that layer are 0 (inactive).
This can be good or bad, depending on your goal.

The output in brackets can be also be changed in the .monitors property of a network.


Using a trained network
=======================

Assuming there are training and valid datasets (train, valid), a network can 
be created and trained in the following way.

#Create an autoencoder
e = theanets.Experiment(theanets.Autoencoder,                                   
                        layers=(784, 150, 784),                                 
                        tied_weights=True,                                      
                        optimize="sgd",                                         
                        )                                                       

#Train the autoencoder, occaisionally checking against the validataion dataset
e.run(train, valid)                                                             
                                                               
#Use the network trained above to encode both training sets.
#The .forward() result holds outputs for each layer in [784, 150, 784]
#Therefore, we can access the code layer results (150) with either [1] or [-2]                 
encoded_train = e.network.forward(train)[-2]                                    
encoded_valid = e.network.forward(valid)[-2]                                    
