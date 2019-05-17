import tensorflow as tf
from baselines.common.models import get_network_builder


class Model(tf.keras.Model):
    def __init__(self, name, network='mlp', **network_kwargs):
        super(Model, self).__init__(name=name)
        self.network_builder = get_network_builder(network)(**network_kwargs)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'layer_normalization' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.nb_actions = nb_actions
        self.output_layer = tf.keras.layers.Dense(units=self.nb_actions,
                                                  activation=tf.keras.activations.tanh,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

    def call(self, obs):
        return self.output_layer(self.network_builder(obs))


class Critic(Model):
    def __init__(self, name='critic', network='mlp', **network_kwargs):
        super().__init__(name=name, network=network, **network_kwargs)
        self.layer_norm = True
        self.output_layer = tf.keras.layers.Dense(units=1,
                                                  kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3),
                                                  name='output')

    def call(self, inputs):
        obs, action = inputs
        x = tf.concat([obs, action], axis=-1)
        x = self.network_builder(x)
        return self.output_layer(x)

    @property
    def output_vars(self):
        return self.output_layer.trainable_variables
