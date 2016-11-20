# neural-network
neural network project for the IPHPC class

References

Easy parallel loops in Python, R, Matlab and Octave
https://blog.dominodatalab.com/simple-parallelization/


An introduction to parallel programming using Python's multiprocessing module
http://sebastianraschka.com/Articles/2014_multiprocessing.html


How to do parallel programming in Python [duplicate]
http://stackoverflow.com/questions/20548628/how-to-do-parallel-programming-in-python


Multiprocessing in Python: a guided tour with examples
section 3:  IPython parallel computing
http://www.davekuhlman.org/python_multiprocessing_01.html#ipython-parallel-computing


Parallel programming with Python's multiprocessing library
Software Carpentry
https://philipwfowler.github.io/2015-01-13-oxford/intermediate/python/04-multiprocessing.html


Parallel Processing and Multiprocessing in Python
https://wiki.python.org/moin/ParallelProcessing

source of MNIST data in csv format:
http://pjreddie.com/projects/mnist-in-csv/

The format is:
label, pix-11, pix-12, pix-13, ...
where pix-ij is the pixel in the ith row and jth column.

---
timing stats


# timing stats:  training data set has 5000 images
# each image is 20x20 = 400 pixels
# t1, t2, t3 are times in seconds

epoch_cnt = c(1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

t1 = c(5.55, 5.79, 12.19, 26.62, 67.83, 140.06, 294.02, 616.55, 1359.42, 2850.34)
t2 = c(2.93, 5.16, 11.29, 25.86, 60.24, 127.71, 273.54, 555.79, 1329.80, 3232.02)
t3 = c(3.01, 5.20, 11.00, 24.27, 55.19, 123.24, 285.96, 583.98, 1345.39)

a1 = c(.2466, .2068, .5798, .6784, .7850, .8902, .9192, .9344, .9462, .9520)
a2 = c(.1638, .3308, .6236, .6830, .8064, .8882, .9192, .9368, .9460, .9558)
a3 = c(.1058, .2910, .4778, .6856, .8152, .8904, .9192, .9346, .9480, .9532)


par(mfrow=c(1,1))
plot(x=epoch_cnt,
     y=t1,
     type="o",
     pch=1,
     xlim = c(0, 525),
     ylim=c(0, 3600),
     main = "compute time vs number of epochs",
     xlab = "number of epochs",
     ylab = "compute time (sec)")

lines(x=epoch_cnt, y=t2, type="p", pch=3)
lines(x=epoch_cnt, y=t3, type="p", pch=5)

plot(x=epoch_cnt,
     y=a1,
     type="o",
     pch=1,
     xlim = c(0, 525),
     ylim=c(0, 1),
     main = "training accuracy vs number of epochs",
     xlab = "number of epochs",
     ylab = "training accuracy")
lines(x=epoch_cnt, y=a2, type="p", pch=3)
lines(x=epoch_cnt, y=a3, type="p", pch=5)
lines(x=epoch_cnt, y=1-a3, type="p", pch=7)

plot(x=epoch_cnt,  y=1-a1, type="p", pch=1, ylim=c(0.02, 0.20), ylab = "training error")
lines(x=epoch_cnt, y=1-a2, type="p", pch=3)
lines(x=epoch_cnt, y=1-a3, type="p", pch=5)
