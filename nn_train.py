
import numpy as np
import scipy.special as ss

k = 10                                 # number of classes
m = 5000                               # number of training examples
n1 = 400                               # pixels per training image

x = np.zeros((m, n1), dtype=float)     # holds all training data
y = np.zeros((m, k), dtype=int)        # holds all labels

def read_x(x, filename, m, n1):
    """ read in the training images (i.e., the x data)
    """
    f = open(filename)

    i = 0
    for line in f:
        tmp = line.strip().split(',')
        x[i, :] = np.array(tmp, dtype=float)
        i += 1
        # end loop

    f.close()
    assert x.shape == (m, n1), "error reading training images"

def read_y(y, filename, m, k):
    """read in the labels (i.e., the y data)
    """
    f = open(filename)

    i = 0
    for line in f:
        tmp = int(line.strip())
        if tmp == 10:
            tmp = 0
        y[i, tmp] = 1
        i += 1

    f.close()
    assert y.shape == (m, k), "error reading training labels"

read_x(x, '../../input-data-nn/x.csv', m, n1)
print(x.shape, np.sum(x))              # temp printing statement

read_y(y, '../../input-data-nn/y.csv', m, k)
print(y.shape, np.sum(y))              # temp printing statement

# initialize constants
# initialize data structures that define the neural network
# import training data: x, y
# initialize weights (once per program)
# epoch loop

#   mini-batch loop

#       initialize cost, gradient for this mini-batch
#       for each training example in this mini-batch...
#           Part 1:  feed-forward network
#           Part 2:  error at Output Layer
#           Part 3:  error at Hidden Layer
#           Part 4:  accumulate gradient, cost info
#       ...next training example
#       Part 5:  finalize gradient, cost calcs
#       Part 6:  export cost for this mini-batch
#       Part 7:  update weight matrices

#   ...next mini-batch

# next epoch
# export weight matrices

## this line was added after attempting to delete from GitHub
