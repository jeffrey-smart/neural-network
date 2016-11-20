
# TODO:  divide cost, gradient by size of mini-batch, not size of input data!!
# I think cost should be independent of number of mini-batches

import numpy as np
import scipy.special as ss
import time

t_start = time.process_time()

## initialize constants
cost = 0                               # cost => J(Theta) in notes
k = 10                                 # number of classes
learn_rate = 0.500                     # learning rate, to update weight matrices
m = 5000                               # number of training examples
n1 = 400                               # pixels per training image
n2 = 25                                # number of nodes in Hidden Layer (not counting bias node)
n3 = k                                 # number of nodes (classes) in the Output Layer

ep_cnt = 512                             # epoch count (number of passes over the training data set
mb_cnt = 10                            # number of mini-batches
td_cnt = m // mb_cnt                   # number of training data points per mini-batch

## initialize data structures that define the neural network
# training data (input data)
x = np.zeros((m, n1), dtype=float)               # holds all training data
y = np.zeros((m, k), dtype=int)                  # holds all labels

# neural network - Input Layer
a1 = np.zeros(n1+1, dtype=float)                 # '+1' for bias node

theta1 = np.zeros((n2, n1+1), dtype = float)     # weights: Input to Hidden Layer
grad1  = np.zeros((n2, n1+1), dtype = float)     # gradient of cost function
mask1  = np.zeros((n2, n1+1), dtype = int)       # used to calculate regularization term

# neural network - Hidden Layer
z2 = np.zeros(n2+1, dtype=float)                 # values; '+1' for bias node (not used)
a2 = np.zeros(n2+1, dtype=float)                 # activations; '+1' for bias node
delta2 = np.zeros(n2+1, dtype=float)             # errors; '+1' for bias node (not used)

theta2 = np.zeros((n3, n2+1), dtype=float)       # weights: Hidden to Output Layer
grad2 =  np.zeros((n3, n2+1), dtype=float)       # gradient of cost function
mask2 =  np.zeros((n3, n2+1), dtype=int)         # used to calculate regularization term

# neural network - Output Layer
z3 = np.zeros(n3, dtype=float)                   # values (no bias node)
a3 = np.zeros(n3, dtype=float)                   # activations
delta3 = np.zeros(n3, dtype=float)               # errors

idx = np.array(m, dtype=int)

## initialize the weights: theta1 and theta2     # one-time initialization
theta1 = np.random.uniform(low=-0.1, high=0.1, size=(n2, n1+1))
theta2 = np.random.uniform(low=-0.1, high=0.1, size=(n3, n2+1))

## initialize the mask (first column is zero; all other columns are ones
mask1[:, 0]  = 0
mask1[:, 1:] = 1
mask2[:, 0]  = 0
mask2[:, 1:] = 1

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

read_y(y, '../../input-data-nn/y.csv', m, k)

# add bias node to training data
x = np.insert(x, 0, 1, axis=1)

## loop over training data
for ep_idx in range(ep_cnt):
    # randomize training data once per epoch
    idx = np.arange(m)
    np.random.shuffle(idx)

    for mb_idx in range(mb_cnt):
        cost = 0.0
        grad1 *= 0                               # initialize grad1
        grad2 *= 0                               # initialize grad2
        for td_idx in range(td_cnt):
            i = idx[ mb_idx * td_cnt + td_idx ]

            # part 1: forward prop
            a1 = x[i, :]                         # a1: 401 x 1

            z2[1:] = np.dot(theta1, a1)          # z2: 26 x 1
            a2[1:] = ss.expit(z2[1:])            # a2: 26 x 1
            a2[0] = 1
            z3 = np.dot(theta2, a2)
            a3 = ss.expit(z3)

            # part 2: error in Output Layer
            delta3 = a3 - y[i, :]

            # part 3: error in Hidden Layer
            delta2 = np.dot(np.transpose(theta2), delta3) * ss.expit(z2) * (1 - ss.expit(z2))
            delta2[0] = np.nan

            # part 4:  accumulate cost and gradient
            cost  -= np.dot(y[i, :], np.log(a3)) + np.dot((np.ones(k) - y[i, :]), np.log(np.ones(k) - a3))
            grad1 += np.outer(delta2[1:], np.transpose(a1))
            grad2 += np.outer(delta3, np.transpose(a2))

        # print intermediate results
        # print(a3)
        # print(y[i,:])

        # part 5:  obtain cost and gradients for this mini-batch
        cost = cost / td_cnt + \
               learn_rate / (2 * td_cnt) * np.sum(mask1 * np.square(theta1)) + \
               learn_rate / (2 * td_cnt) * np.sum(mask2 * np.square(theta2))

        # print(ep_idx, mb_idx, cost)
        # print('{:4d} {:4d} {:6.3f}'.format(ep_idx, mb_idx, cost))

        grad1 = grad1 / td_cnt + (learn_rate / td_cnt) * theta1 * mask1
        grad2 = grad2 / td_cnt + (learn_rate / td_cnt) * theta2 * mask2

        # part 6: update weights
        theta1 -= learn_rate * grad1
        theta2 -= learn_rate * grad2

        # TODO: export weight vectors for later use

        # print some predictions
        # print(" -- prediction --")
        # print(a3)

# TODO:  make prediction for entire input data; compare label vs. prediction

# initialize constants - DONE
# initialize data structures that define the neural network - DONE
# import training data: x, y - DONE
# initialize weights (once per program) - DONE
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

# TODO - export weight matrices

a1 = x;
print(x.shape, " = x.shape")
print(a1.shape, " = a1.shape")

z2 = np.dot(a1, np.transpose(theta1))
print(z2.shape, " = z2.shape")

a2 = ss.expit(z2)
a2 = np.insert(a2, 0, 1, axis=1)
print(a2.shape, " = a2.shape")

z3 = np.dot(a2, np.transpose(theta2))
print(z3.shape, " = z3.shape")

a3 = ss.expit(z3)
print(a3.shape, " = a3.shape")

actual_idx = np.argmax(y, axis=1)
print(actual_idx.shape, " = actual_idx.shape")

predict_idx = np.argmax(a3, axis=1)
print(predict_idx.shape, " predict_idx.shape")

correct = np.count_nonzero(actual_idx == predict_idx)
print(correct, " = number of correct in training analysis")
print(correct / m, " = training accuracy")
print(ep_cnt, " = number of epochs")

t_end = time.process_time()
print("time = ", t_end - t_start)
