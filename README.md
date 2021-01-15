# lyapunov-hyperopt
The lyapunov.py module contains the code used to calculate Jacobians and lyapunov exponents. The code is written primarily using Pytorch

The primary methods to look at in that module are calc_LEs_an and LE_stats. The first calculates the exponents for given network with a batch of inputs. These results can be passed to LE_stats in order to find the mean and standard deviation of the exponents over that batch.


Configuration classes are used to track hyper-parameters of data, training, and models. See config.py for details. Hyperparameters of Lyapunov Exponent calculation are set and tracked in the same way.