from copy import copy
from functools import reduce

import numpy as np
import tensorflow as tf

from baselines import logger
from baselines.ddpg.models import Actor, Critic
from baselines.common.mpi_running_mean_std import RunningMeanStd
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / (stats.std + 1e-8)


def denormalize(x, stats):
    if stats is None:
        return x
    return x * stats.std + stats.mean

def reduce_std(x, axis=None, keepdims=False):
    return tf.sqrt(reduce_var(x, axis=axis, keepdims=keepdims))

def reduce_var(x, axis=None, keepdims=False):
    m = tf.reduce_mean(x, axis=axis, keepdims=True)
    devs_squared = tf.square(x - m)
    return tf.reduce_mean(devs_squared, axis=axis, keepdims=keepdims)

def update_perturbed_actor(actor, perturbed_actor, param_noise_stddev):
    assert len(actor.variables) == len(perturbed_actor.variables)
    assert len(actor.perturbable_vars) == len(perturbed_actor.perturbable_vars)

    for var, perturbed_var in zip(actor.variables, perturbed_actor.variables):
        if var in actor.perturbable_vars:
            # logger.info('  {} <- {} + noise'.format(perturbed_var.name, var.name))
            perturbed_var.assign(var + tf.random.normal(shape=tf.shape(var), mean=0., stddev=param_noise_stddev))
        else:
            # logger.info('  {} <- {}'.format(perturbed_var.name, var.name))
            perturbed_var.assign(var)


class DDPG(object):
    def __init__(self, actor, critic, memory, observation_shape, action_shape, param_noise=None, action_noise=None,
        gamma=0.99, tau=0.001, normalize_returns=False, enable_popart=False, normalize_observations=True,
        batch_size=128, observation_range=(-5., 5.), action_range=(-1., 1.), return_range=(-np.inf, np.inf),
        critic_l2_reg=0., actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.):

        # Parameters.
        self.gamma = gamma
        self.tau = tau
        self.memory = memory
        self.normalize_observations = normalize_observations
        self.normalize_returns = normalize_returns
        self.action_noise = action_noise
        self.param_noise = param_noise
        self.action_range = action_range
        self.return_range = return_range
        self.observation_range = observation_range
        self.observation_shape = observation_shape
        self.critic = critic
        self.actor = actor
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.clip_norm = clip_norm
        self.enable_popart = enable_popart
        self.reward_scale = reward_scale
        self.batch_size = batch_size
        self.stats_sample = None
        self.critic_l2_reg = critic_l2_reg

        # Observation normalization.
        if self.normalize_observations:
            with tf.name_scope('obs_rms'):
                self.obs_rms = RunningMeanStd(shape=observation_shape)
        else:
            self.obs_rms = None

        # Return normalization.
        if self.normalize_returns:
            with tf.name_scope('ret_rms'):
                self.ret_rms = RunningMeanStd()
        else:
            self.ret_rms = None

        # Create target networks.
        self.target_actor = Actor(actor.nb_actions, observation_shape, name='target_actor', network=actor.network, **actor.network_kwargs)
        self.target_critic = Critic(actor.nb_actions, observation_shape, name='target_critic', network=critic.network, **critic.network_kwargs)

        # Set up parts.
        if self.param_noise is not None:
            self.setup_param_noise()

        self.actor_optimizer = tf.keras.optimizers.Adam()
        self.critic_optimizer = tf.keras.optimizers.Adam()

        if self.normalize_returns and self.enable_popart:
            self.setup_popart()

        self.initial_state = None # recurrent architectures not supported yet


    def setup_param_noise(self):
        assert self.param_noise is not None

        # Configure perturbed actor.
        self.perturbed_actor = Actor(self.actor.nb_actions, self.observation_shape, name='param_noise_actor', network=self.actor.network, **self.actor.network_kwargs)

        # Configure separate copy for stddev adoption.
        self.perturbed_adaptive_actor = Actor(self.actor.nb_actions, self.observation_shape, name='adaptive_param_noise_actor', network=self.actor.network, **self.actor.network_kwargs)

    def setup_popart(self):
        # See https://arxiv.org/pdf/1602.07714.pdf for details.
        for vs in [self.critic.output_vars, self.target_critic.output_vars]:
            assert len(vs) == 2
            M, b = vs
            assert 'kernel' in M.name
            assert 'bias' in b.name
            assert M.get_shape()[-1] == 1
            assert b.get_shape()[-1] == 1


    def step(self, obs, apply_noise=True, compute_Q=True):
        normalized_obs = tf.clip_by_value(normalize(obs, self.obs_rms), self.observation_range[0], self.observation_range[1])
        if self.param_noise is not None and apply_noise:
            action = self.perturbed_actor(normalized_obs)
        else:
            action = self.actor(normalized_obs)

        if compute_Q:
            normalized_critic_with_actor_tf = self.critic(normalized_obs, action)
            q = denormalize(tf.clip_by_value(normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        else:
            q = None

        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == action[0].shape
            action += noise
        action = np.clip(action, self.action_range[0], self.action_range[1])


        return action, q, None, None

    def store_transition(self, obs0, action, reward, obs1, terminal1):
        reward *= self.reward_scale

        B = obs0.shape[0]
        for b in range(B):
            self.memory.append(obs0[b], action[b], reward[b], obs1[b], terminal1[b])
            if self.normalize_observations:
                self.obs_rms.update(np.array([obs0[b]]))

    @tf.function
    def train(self, obs0, obs1, actions, rewards, terminals1):
        # Get a batch.
        normalized_obs1 = tf.clip_by_value(normalize(obs1, self.obs_rms), self.observation_range[0], self.observation_range[1])
        Q_obs1 = denormalize(self.target_critic(normalized_obs1, self.target_actor(normalized_obs1)), self.ret_rms)
        target_Q = rewards + (1. - terminals1) * self.gamma * Q_obs1

        if self.normalize_returns and self.enable_popart:
            old_mean = self.ret_rms.mean
            old_std = self.ret_rms.std
            self.ret_rms.update(target_Q.flatten())
            # renormalize Q outputs
            new_mean = self.ret_rms.mean
            new_std = self.ret_rms.std
            for vs in [self.critic.output_vars, self.target_critic.output_vars]:
                kernel, bias = vs
                kernel.assign(kernel * old_std / new_std)
                bias.assign((bias * old_std + old_mean - new_mean) / new_std)


        normalized_obs0 = tf.clip_by_value(normalize(obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        with tf.GradientTape() as tape:
            actor_tf = self.actor_tf = self.actor(normalized_obs0)
            normalized_critic_tf = self.critic(normalized_obs0, actions)
            critic_tf = denormalize(tf.clip_by_value(normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
            normalized_critic_with_actor_tf = self.critic(normalized_obs0, actor_tf)
            critic_with_actor_tf = denormalize(tf.clip_by_value(normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
            actor_loss = -tf.reduce_mean(critic_with_actor_tf)
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads_and_vars = zip(actor_grads, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(actor_grads_and_vars)


        with tf.GradientTape() as tape:
            normalized_critic_tf = self.critic(normalized_obs0, actions)
            normalized_critic_target_tf = tf.clip_by_value(normalize(target_Q, self.ret_rms), self.return_range[0], self.return_range[1])
            critic_loss = tf.reduce_mean(tf.square(normalized_critic_tf - normalized_critic_target_tf))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads_and_vars = zip(critic_grads, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(critic_grads_and_vars)

        return critic_loss, actor_loss

    def initialize(self):
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

    def update_target_net(self):
        for var, target_var in zip(self.actor.variables, self.target_actor.variables):
            target_var.assign((1. - self.tau) * target_var + self.tau * var)
        for var, target_var in zip(self.critic.variables, self.target_critic.variables):
            target_var.assign((1. - self.tau) * target_var + self.tau * var)

    def get_stats(self):

        if self.stats_sample is None:
            # Get a sample and keep that fixed for all further computations.
            # This allows us to estimate the change in value for the same set of inputs.
            self.stats_sample = self.memory.sample(batch_size=self.batch_size)
        obs0 = self.stats_sample['obs0']
        actions = self.stats_sample['actions']
        normalized_obs0 = tf.clip_by_value(normalize(obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        normalized_critic_tf = self.critic(normalized_obs0, actions)
        critic_tf = denormalize(tf.clip_by_value(normalized_critic_tf, self.return_range[0], self.return_range[1]), self.ret_rms)
        actor_tf = self.actor(normalized_obs0)
        normalized_critic_with_actor_tf = self.critic(normalized_obs0, actor_tf)
        critic_with_actor_tf = denormalize(tf.clip_by_value(normalized_critic_with_actor_tf, self.return_range[0], self.return_range[1]), self.ret_rms)

        stats = {}
        if self.normalize_returns:
            stats['ret_rms_mean'] = self.ret_rms.mean
            stats['ret_rms_std'] = self.ret_rms.std
        if self.normalize_observations:
            stats['obs_rms_mean'] = tf.reduce_mean(self.obs_rms.mean)
            stats['obs_rms_std'] = tf.reduce_mean(self.obs_rms.std)
        stats['reference_Q_mean'] = tf.reduce_mean(critic_tf)
        stats['reference_Q_std'] = reduce_std(critic_tf)
        stats['reference_actor_Q_mean'] = tf.reduce_mean(critic_with_actor_tf)
        stats['reference_actor_Q_std'] = reduce_std(critic_with_actor_tf)
        stats['reference_action_mean'] = tf.reduce_mean(actor_tf)
        stats['reference_action_std'] = reduce_std(actor_tf)

        if self.param_noise:
            perturbed_actor_tf = self.perturbed_actor(normalized_obs0)
            stats['reference_perturbed_action_mean'] = tf.reduce_mean(perturbed_actor_tf)
            stats['reference_perturbed_action_std'] = reduce_std(perturbed_actor_tf)
            stats.update(self.param_noise.get_stats())
        return stats


    def adapt_param_noise(self):
        try:
            from mpi4py import MPI
        except ImportError:
            MPI = None

        if self.param_noise is None:
            return 0.

        # Perturb a separate copy of the policy to adjust the scale for the next "real" perturbation.
        batch = self.memory.sample(batch_size=self.batch_size)
        update_perturbed_actor(self.actor, self.perturbed_adaptive_actor, self.param_noise.current_stddev)

        obs0 = batch['obs0']
        normalized_obs0 = tf.clip_by_value(normalize(obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        actor_tf = self.actor(normalized_obs0)
        adaptive_actor_tf = self.perturbed_adaptive_actor(normalized_obs0)
        distance = tf.sqrt(tf.reduce_mean(tf.square(actor_tf - adaptive_actor_tf)))


        if MPI is not None:
            mean_distance = MPI.COMM_WORLD.allreduce(distance, op=MPI.SUM) / MPI.COMM_WORLD.Get_size()
        else:
            mean_distance = distance

        self.param_noise.adapt(mean_distance)
        return mean_distance

    def reset(self):
        # Reset internal state after an episode is complete.
        if self.action_noise is not None:
            self.action_noise.reset()
        if self.param_noise is not None:
            update_perturbed_actor(self.actor, self.perturbed_actor, self.param_noise.current_stddev)
