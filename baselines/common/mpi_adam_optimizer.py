import numpy as np
import tensorflow as tf
from baselines.common.tests.test_with_mpi import with_mpi
from baselines import logger
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

class MpiAdamOptimizer(tf.keras.optimizers.Adam):
    """Adam optimizer that averages gradients across mpi processes."""
    def __init__(self, comm, **kwargs):
        self.comm = comm
        super(MpiAdamOptimizer, self).__init__(name='MpiAdam', **kwargs)

    def apply_gradients(self, grads_and_vars):
        grads = [g for g, _ in grads_and_vars if g is not None]
        var_list = [v for g, v in grads_and_vars if g is not None]
        flat_grad = tf.concat([tf.reshape(g, (-1,)) for g in grads], axis=0)
        shapes = [v.shape.as_list() for v in var_list]
        sizes = [int(np.prod(s)) for s in shapes]

        buf = np.zeros(sum(sizes), np.float32)
        self.comm.Allreduce(flat_grad.numpy(), buf, op=MPI.SUM)
        avg_flat_grad = np.divide(buf, float(self.comm.Get_size()))
        avg_grads = tf.split(avg_flat_grad, sizes, axis=0)
        avg_grads_and_vars = []
        for grad, v in zip(avg_grads, var_list):
            avg_grads_and_vars.append((tf.reshape(grad, v.shape), v))
        super(MpiAdamOptimizer, self).apply_gradients(avg_grads_and_vars)
        if self.iterations.numpy() % 100 == 0:
            check_synced(tf.reduce_sum(avg_grads_and_vars[0][1]).numpy())


def check_synced(localval, comm=None):
    """
    It's common to forget to initialize your variables to the same values, or
    (less commonly) if you update them in some other way than adam, to get them out of sync.
    This function checks that variables on all MPI workers are the same, and raises
    an AssertionError otherwise

    Arguments:
        comm: MPI communicator
        localval: list of local variables (list of variables on current worker to be compared with the other workers)
    """
    comm = comm or MPI.COMM_WORLD
    vals = comm.gather(localval)
    if comm.rank == 0:
        assert all(val==vals[0] for val in vals[1:]),\
            'MpiAdamOptimizer detected that different workers have different weights: {}'.format(vals)

@with_mpi(timeout=5)
def test_nonfreeze():
    np.random.seed(0)
    tf.set_random_seed(0)

    a = tf.Variable(np.random.randn(3).astype('float32'))
    b = tf.Variable(np.random.randn(2,5).astype('float32'))
    loss = tf.reduce_sum(tf.square(a)) + tf.reduce_sum(tf.sin(b))

    stepsize = 1e-2
    # for some reason the session config with inter_op_parallelism_threads was causing
    # nested sess.run calls to freeze
    config = tf.ConfigProto(inter_op_parallelism_threads=1)
    sess = U.get_session(config=config)
    update_op = MpiAdamOptimizer(comm=MPI.COMM_WORLD, learning_rate=stepsize).minimize(loss)
    sess.run(tf.global_variables_initializer())
    losslist_ref = []
    for i in range(100):
        l,_ = sess.run([loss, update_op])
        print(i, l)
        losslist_ref.append(l)
