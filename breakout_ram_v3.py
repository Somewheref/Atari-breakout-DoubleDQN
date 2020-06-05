from __future__ import division

import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())


import argparse
import numpy as np
import pandas as pd
import gym
import itertools
import tqdm

from time import time
from collections import deque
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Convolution2D, Permute
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt


# hyperparams
cont_frames_nb = 4      # number of consecutive frames for one state
input_shape = (84,84)
INPUT_SHAPE = (cont_frames_nb, ) + input_shape
BATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 100000     # 1000000 in Nature DQN, approx.= 26GB
REPLAY_START_SIZE =  5000       # 50000 in Nature DQN
LEARNING_RATE = 0.00025
GAMMA = 0.99
EPSILON_DECREASE_RATE = 0.0001    # in nature it is 9e-7
MAX_EPSILON = 1
MIN_EPSILON = 0.1
UPDATE_FREQ = 4
TARGET_NET_UPDATE_FREQ = 500
INITIAL_RANDOM_ACTIONS_NB = 30

# gym environment name
env_name = "Breakout-v0"

# default saving path
default_weights_filename = "models/model_3.h5"

# display visualization or not
#render = False
# changes: if we want to display UI while training,
# use python breakout_v3.py --render
# when in terminal to execute

prewarming = True

# display logs when training
# 0: no logging information
# 1: info on episode ends
# 2: info on every step
verbosity = 1

# average line when plotting the graph
avg_line_update_rate = 5

# the number of episodes to save a weight
save_weights_episodes_nb = 20

log_info_steps = 1000


class Processor():
    """processor for processing images and batches"""
    @staticmethod
    def process_image(image):
        assert image.ndim == 3, 'expecting image of dimension %s, dimension of %s recieved' %('3', image.ndim)  # 3 for (channel, height, weight)
        img = Image.fromarray(image)
        img = img.resize(input_shape).convert('L')  # to grayscale
        img = np.asarray(img.getdata(), dtype=np.uint8).reshape(img.size[1], img.size[0])
        return img

    @staticmethod
    def process_batch(batch_state):
        return batch_state.astype('float32') / 128. - 1.
    
    @staticmethod
    def stack_and_process_frames(previous_frames = None, frame = None):
        frame = Processor.process_image(frame)
        if previous_frames is None:
            return np.array([frame,] * cont_frames_nb)
        return np.append(previous_frames[1:], [frame,], axis=0)     # stacking frames


class DQNRelpayer():
    def __init__(self, capacity):
        self.memory = pd.DataFrame(index = range(capacity),
            columns = ['observation', 'action', 'reward',
                        'next_observation', 'done'])
        self.i = 0
        self.count = 0
        self.capacity = capacity

    def store(self, *args):
        self.memory.loc[self.i] = args
        self.i = (self.i + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def sample(self,size):
        indices = np.random.choice(self.count, size = size)
        batch = [np.stack(self.memory.loc[indices, field]) for field in self.memory.columns]
        batch[0] = Processor.process_batch(batch[0])
        batch[3] = Processor.process_batch(batch[3])
        return batch



class DQNAgent():
    """
    an agent that uses double Q learning
    but first let us use Q learning
    """
    def __init__(self, env, gamma = 0.99, start_eps = 1, end_eps = 0.001,
                replayer_capacity = 1000000, batch_size = 64, update_freq = 4,
                target_update_freq = 10000, **kwargs):
        observation_dim = env.observation_space.shape[0]
        self.action_n = env.action_space.n
        self.gamma = gamma
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.epsilon = start_eps
        self.batch_size = batch_size
        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        
        self.replayer = DQNRelpayer(replayer_capacity)
        self.env = env

        # getting from kwargs
        self.replay_start_size = kwargs.get("replay_start_size", 50000)
        self.clip_rewards = kwargs.get("clip_rewards", False)
        self.epsilon_decrease_rate = kwargs.get("epsilon_decrease_rate", 9e-7)

        # keep track of which step we are
        self.step = 0
        self.fit_count = 0

        # create models
        self.evalutate_net = self.build_net()
        self.target_net = self.build_net()

        # load model
        # prevent the folder "model" is not in current dir (first train)
        if weights_filename == default_weights_filename and not os.path.isdir('model/'):
            os.system('mkdir model')
        if os.path.isfile(weights_filename):
            print('loading weights from file:' + weights_filename)
            self.evalutate_net.load_weights(weights_filename)
            self.target_net.load_weights(weights_filename)
        
        # percentage bar
        self.pbar = tqdm.tqdm(total = self.replay_start_size)
        self.pbar.set_description("prewarming... ")


    def build_net(self, callbacks = None):
        # model of Nature DQN
        model = Sequential([
            Permute((2,3,1), input_shape = INPUT_SHAPE),
            Convolution2D(32, (8, 8), strides = (4, 4), activation = 'relu'),
            Convolution2D(64, (4, 4), strides = (2, 2), activation = 'relu'),
            Convolution2D(64, (3, 3), strides = (1, 1), activation = 'relu'),
            Flatten(),
            Dense(512, activation = 'relu'),
            Dense(self.action_n, activation='linear')
        ])
        if verbosity >= 1:
            print(model.summary())
        optimizer = Adam(lr=.0005)
        if not callbacks:
            callbacks = self.get_callbacks()
        model.compile(loss = 'mse', optimizer = optimizer)
        return model

    
    def decide(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_n)
        else:
            q_values = self.evalutate_net.predict(state[np.newaxis])
            action = np.argmax(q_values)
        return action

    
    def learn(self, state, action, reward, next_state, done):
        # store the experience (several frames)
        self.replayer.store(state, action, reward, next_state, done)
        self.step += 1

        # prewarm percentage bar
        if self.replayer.count <= self.replay_start_size:
            self.pbar.update(1)
            if self.replayer.count == self.replay_start_size:
                self.pbar.close()
                global prewarming
                prewarming = False
                print("prewarm done.")

        # does not learn until a few steps or if we do not have enough sample in memory
        if self.step % self.update_freq == 0 and not prewarming:
            # experience replay
            # for some reason I find it cannot be used in for loop directly
            # so I did a few prepocessing
            states, actions, rewards, next_states, dones = self.replayer.sample(self.batch_size)
            if self.clip_rewards:
                reward = np.clip(reward, -1, 1)

            next_qs = self.target_net.predict(next_states)
            next_max_ps = next_qs.max(axis = -1)

            target_qs = self.evalutate_net.predict(states)
            target_qs[range(self.batch_size), actions] = \
                rewards + self.gamma * next_max_ps * (1 - dones)
            #predictions = self.evalutate_net.predict(states)
            #max_qs = predictions.max(axis = -1)
            #predictions[range(self.batch_size), actions] = rewards + self.gamma * max_qs * (1 - dones)
            #self.evalutate_net.fit(states, predictions, verbose = 0, epochs = 1)
            self.evalutate_net.fit(states, target_qs, verbose = 0)
            self.fit_count += 1

            if self.fit_count % self.target_update_freq == 0:
                #transfer the weights of evalutate net to target net
                if verbosity >= 2:
                    print("*" * 30)
                    print("updating target network...")
                    print("*" * 30)

                self.target_net.set_weights(self.evalutate_net.get_weights())
                
            # annealing epsilon
            if self.step > self.replay_start_size:
                self.epsilon = max(self.epsilon - self.epsilon_decrease_rate, self.end_eps)
            
            if verbosity >= 2:
                print("step: %s, reward: %s, epsilon: %s" %(self.step, reward, round(self.epsilon, 4)))
            
            if verbosity >= 1 and self.step % log_info_steps == 0:
                print('-' * 30)
                print('INFO: step: %s, epsilon: %s' %(self.step, self.epsilon))
                print('-' * 30)

    def get_callbacks(self):
        # tensorboard
        # type tensorboard --logdir=log/    to visualize
        #tensorboard = TensorBoard(log_dir = 'logs/{}'.format(time()))
        #return [tensorboard,]
        return []

    
    def save(self, filename = None):
        if filename is None:
            self.evalutate_net.save(weights_filename)
        else:
            self.evalutate_net.save(filename)


if __name__ == "__main__":
    # Argparser, so that we can quickly select options in terminal
    # without the need to open the code every time
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices = ['train','test'], default = 'train')
    parser.add_argument('--weights', type = str, default = default_weights_filename)
    parser.add_argument("--render", action='store_true', help='render or not')

    p_args = parser.parse_args()
    training = True
    if p_args.mode == 'test':
        training = False
    weights_filename = p_args.weights

    render = p_args.render

    # set a seed so that we may perform the same results
    np.random.seed(666)

    # construct environment
    env = gym.make(env_name)
    env.seed(666)
    agent = DQNAgent(env,GAMMA,start_eps=MAX_EPSILON,end_eps=MIN_EPSILON,replayer_capacity=REPLAY_MEMORY_SIZE,
            batch_size=BATCH_SIZE, update_freq=UPDATE_FREQ,target_update_freq=TARGET_NET_UPDATE_FREQ,
            replay_start_size = REPLAY_START_SIZE)
    
    history_episode_rewards = []
    average_episode_rewards = []
    max_epsiode_reward = -999999
    try:
        for episode in itertools.count():
            epsiode_reward = 0
            state = env.reset()
            state = Processor.stack_and_process_frames(None, state)

            for step in itertools.count():
                if render:
                    env.render()
                if step <= INITIAL_RANDOM_ACTIONS_NB:
                    action = np.random.choice(range(env.action_space.n))  # default action
                else:
                    # agent take action
                    action = agent.decide(Processor.process_batch(state))
                observation, reward, done, _ = env.step(action)
                epsiode_reward += reward
                observation = Processor.stack_and_process_frames(state, observation)
                agent.learn(state, action, reward, observation, done)
                state = observation

                if done:
                    if epsiode_reward > max_epsiode_reward and not prewarming:
                        if verbosity >= 1:
                            print('*' * 40)
                            print('saving model of episode reward %s' %epsiode_reward)
                            print('*' * 40)
                        agent.save()    # save model on episode ends
                        max_epsiode_reward = epsiode_reward
                    break
            if not prewarming:
                history_episode_rewards.append(epsiode_reward)
                if episode % avg_line_update_rate == 0 and episode != 0:
                    average_episode_rewards.append(sum(history_episode_rewards) / episode)
                # save model
                if episode % save_weights_episodes_nb == 0:
                    agent.save("model/weights_{}.h5".format(episode))
            if verbosity >= 1 and not prewarming:
                print("episode: %s, reward: %s" %(episode, epsiode_reward))

    finally:
        print("=" * 40)
        print("training session complete...")
        print("=" * 40)
        # display graph
        plt.figure(figsize=(8,4))
        plt.xlabel('episodes')
        plt.ylabel('rewards')
        plt.plot(range(len(history_episode_rewards)),history_episode_rewards)
        plt.plot(np.array(range(len(average_episode_rewards))) * avg_line_update_rate, average_episode_rewards)
        plt.show()

        is_to_save = input("save weights? (y/n)")
        if is_to_save in ('Y', 'y'):
            print("saving weights...")
            agent.save()
            print("save complete.")
        else:
            print("weights discarded.")
