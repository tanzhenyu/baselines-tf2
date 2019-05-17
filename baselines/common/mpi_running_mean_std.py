try:
    from mpi4py import MPI
except ImportError:
    MPI = None

import tensorflow as tf, numpy as np

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-2, shape=()):

        self._sum = tf.Variable(
            initial_value=np.zeros(shape=shape, dtype=np.float64),
            dtype=tf.float64,
            name="runningsum")
        self._sumsq = tf.Variable(
            initial_value=np.full(shape=shape, epsilon, dtype=np.float64),
            dtype=tf.float64,
            name="runningsumsq")
        self._count = tf.Variable(
            initial_value=0.,
            dtype=tf.float64,
            name="count")
        self.shape = shape
        self.epsilon = epsilon

    def update(self, x):
        x = x.astype('float64')
        n = int(np.prod(self.shape))
        totalvec = np.zeros(n*2+1, 'float64')
        addvec = np.concatenate([x.sum(axis=0).ravel(), np.square(x).sum(axis=0).ravel(), np.array([len(x)],dtype='float64')])
        if MPI is not None:
            MPI.COMM_WORLD.Allreduce(addvec, totalvec, op=MPI.SUM)
        self.incfiltparams(totalvec[0:n].reshape(self.shape), totalvec[n:2*n].reshape(self.shape), totalvec[2*n])

    @property
    def mean(self):
        return tf.cast(self._sum / self._count, tf.float32)

    @property
    def std(self):
        return tf.sqrt(tf.maximum(tf.cast(self._sumsq / self._count, tf.float32) - tf.square(self.mean), self.epsilon))