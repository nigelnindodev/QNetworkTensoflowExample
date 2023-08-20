from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay
import reverb

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

# Set up a virtual display for rendering OpenAI gym environments.
display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

# Define model hyperparameters
num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

# Load CartPole environment
env_name = 'CartPole-v0'
env = suite_gym.load(env_name)

env.reset()
PIL.Image.fromarray(env.render())


# env.step() takes an 'action' in the environment and returns a 'TimeStep' tuple containing the next of the environment and the reward for the action
# time_step_spec() returns the specification of the 'TimeStep' tuple

# 'observation' attribute shows the shape of the observations, data type, and ranges of allowed values
print('Observation Spec:')
print(env.time_step_spec().observation)

# 'reward' shows the details of the reward
print('Reward Spec:')
print(env.time_step_spec().reward)

# action_spec() returns the shape, data type and allowed values of valid actions
print('Action Spec:')
print(env.action_spec())

"""
From the data, we can see that for the cartpole environment:

observation: An array of 4 floats
    - the position and velocity of the cart
    - the angular position and velocity of the pole

reward: A scala float value

action: Scala integer with two possible values:
    - 0: move left
    - 1: move right
"""

time_step = env.reset()
print('Time step:')
print(time_step)

# move right
action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('Next time step:')
print(next_time_step)

# instantiate two environments, training and evaluate. This is usually the norm.
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

# use 'TFPyEnvironment' to convert the environments to 'Tensors'
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


"""
For this example we will use a DQN agent.

This uses a 'QNetwork', that will be used to predict 'QValues (expected return for all actions, given observations from the environment)'.
"""


fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])


# Instantiate a 'DqnAgent'

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

"""
We now need to define a policy, which is the way an agent acts in an environment. 

The policy returns an action (left or right for each time step observation) that helps keep the cartpole upright.

Agents contain two policies:
    - agent.policy: Main policy used for evaluation and deployment.
    - agent.collect_policy: A second policy used for data collection.
"""

eval_policy = agent.policy
collect_policy = agent.collect_policy

"""
Policies can be created independently for the agents.

For example, the policy below policy randomly selects an action for each time step.
"""
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

"""
To get an action from a policy, call the policy.action(time_step) method. The time_step contains the observation from the environment.

This method returns a 'PolicyStep', which is a named tuple with 3 components:

action: Action to be takes, in this case 1 or 0
state: Used for stateful policies (such as one based on a RNN)
info: auxilliary data such as log probabilities of actions
"""

# crete a new environment and test the random policy

example_environment = tf_py_environment.TFPyEnvironment(
    suite_gym.load('CartPole-v0'))

time_step = example_environment.reset()
random_policy.action(time_step)

"""
Most common metric used to evaluate a policy is the average return. This is the sum of rewards obtained when running a policy in
an environment for an episode.

The following function computes the average return of a policy, given the policy, environment, and a number of episodes.
"""

def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]

print("Average return for random policy: ")
print(compute_avg_return(eval_env, random_policy, num_eval_episodes))

"""
In order to keep track of the data collected from the environment, we will use Reverb.

It will store data when we get trajectories, and is consumed during training.

Te replay buffer is constructed using the specs describing the tensors that are to be stored, which can be retrieved from the agent
using agent.collect_data_spec
"""

table_name = 'uniform_table'
replay_buffer_signature = tensor_spec.from_spec(
      agent.collect_data_spec)
replay_buffer_signature = tensor_spec.add_outer_dim(
    replay_buffer_signature)

table = reverb.Table(
    table_name,
    max_size=replay_buffer_max_length,
    sampler=reverb.selectors.Uniform(),
    remover=reverb.selectors.Fifo(),
    rate_limiter=reverb.rate_limiters.MinSize(1),
    signature=replay_buffer_signature)

reverb_server = reverb.Server([table])

replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
    agent.collect_data_spec,
    table_name=table_name,
    sequence_length=2,
    local_server=reverb_server)

rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
  replay_buffer.py_client,
  table_name,
  sequence_length=2)

"""
For most agents, collect_data_spec is a named tuple called Trajectory, containing the specs for observations, actions, rewards, and other items.
"""
print("agent collect_data_spec: ")
print(agent.collect_data_spec)

print("agent collect_data_spec fields: ")
print(agent.collect_data_spec._fields)

"""
We can now execute the random policy, recording the data in the replay buffer. 
"""

py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      random_policy, use_tf_function=True),
    [rb_observer],
    max_steps=initial_collect_steps).run(train_py_env.reset())

"""
The agent also now needs to access the replay buffer. This is provided by creating an iterable tf.data.Dataset pipeline which will feed data to the agent.

Each row of the replay buffer only stores a single observation step. 
But since the DQN Agent needs both the current and next observation to compute the loss, the dataset pipeline will sample two adjacent rows for each item in the batch (num_steps=2).
"""
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3,
    sample_batch_size=batch_size,
    num_steps=2).prefetch(3)

print("Driver trajectory dataset: ")
print(dataset)

# create an iterator on the dataset
iterator = iter(dataset)
print("First iterator values: ")
print(iterator.next())

"""
We are now ready to train the agent. The average return should increase.
"""

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step.
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

# Reset the environment.
time_step = train_py_env.reset()

# Create a driver to collect experience.
collect_driver = py_driver.PyDriver(
    env,
    py_tf_eager_policy.PyTFEagerPolicy(
      agent.collect_policy, use_tf_function=True),
    [rb_observer],
    max_steps=collect_steps_per_iteration)

for _ in range(num_iterations):

  # Collect a few steps and save to the replay buffer.
  time_step, _ = collect_driver.run(time_step)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)




