from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ems_env import ems

# from networks import evaluator, forecaster

EPISODES = 500
PRINT_EVERY_X_ITER = 10
EPISODE_LENGTH = 480
SEARCH_DEPTH = 40
MEMORY_LENGTH = 50000

from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
import keras.backend as K
import tensorflow as tf

def evaluator_net(obs=2, actions=3):
    inp = Input((obs,))
    x = Dense(24, activation='relu')(inp)
    # x = Dense(24, activation='relu')(x)
    x_val = Dense(24, activation='relu')(x)
    # x_act = Dense(24, activation='relu')(x)
    # out0 = Dense(actions, activation='softmax', name="actor_head")(x_act)
    out1 = Dense(actions, activation="linear", name="value_head")(x_val)

    # m = Model(inp, [out0, out1])
    # m.compile(optimizer = SGD(0.0001, momentum = 0.9), loss={"actor_head": "categorical_crossentropy", "value_head": "MSE"})
    m = Model(inp, [out1])
    m.compile(optimizer = SGD(0.0001, momentum = 0.9), loss=_huber_loss)
    return m

def _huber_loss(y_true, y_pred, clip_delta=1.0):
    ### source: keon.io ###
    error = y_true - y_pred
    cond  = K.abs(error) <= clip_delta

    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

    return K.mean(tf.where(cond, squared_loss, quadratic_loss))

def prepare_eval_train(eval_train, evaluator_network):
    states = np.stack(eval_train[:,0])[:,4:]
    actions = np.stack(eval_train[:,2])
    rewards = np.stack(eval_train[:,1])
    x = states

    # eval_df = pd.DataFrame(eval_train)
    # states = pd.DataFrame(np.vstack(eval_df[0]))
    # x = states[states.columns[4:]]
    # # action_priors = pd.DataFrame(np.vstack(eval_df[2])/eval_df[2].sum())
    index = range(x.shape[0])
    # # action_priors = np.zeros((eval_df.shape[0], 3))
    # # action_priors[index, np.vstack(eval_df[2]).reshape(-1)] = 1
    values = evaluator_network.predict(x)
    values[index, actions] = rewards
    return x, [values]

def prepare_forecast_train(forecast_train):
    forecast_df = pd.DataFrame(forecast_train)
    states = pd.DataFrame(np.vstack(forecast_df[0]))
    next_states = pd.DataFrame(np.vstack(forecast_df[1]))
    y = forecaster.prepare_forecast_set(next_states[5])
    x = states[states.columns[0:4]]
    return x, y

def plot_test(evaluator_network, forecast_network, LOGFILE=True, PLOT=False):
    test_env = ems(EPISODE_LENGTH)
    state = test_env.reset()
    test_env.time = 20000
    log, soc = [], []
    cum_r = 0
    for i in range(960):
        action = np.argmax(evaluator_network.predict(np.expand_dims(state[4:], axis=0))[0])
        # action, root = UCT_search(StateNode(state, env.time), SEARCH_DEPTH, evaluator=evaluator_network, forecaster=forecast_network, use_dirichlet=False)
        state, r, done, _ = test_env.step(action)
        log.append([action, state[4], state[5], r])
        soc.append(state[4])
        cum_r += r
    tqdm.write(f" Current weights achieve a score of {cum_r}")
    # if cum_r > self.high_score and self.SAVE_HIGHSCORE:
    #     self.high_score = cum_r
    #     self.actor_target.save_weights(f"high_score_weights_{cum_r}.h5")
    if PLOT:
        pd.DataFrame(soc).plot()
        plt.show(block=True)
        plt.close()
    if LOGFILE:
        xls = pd.DataFrame(log)
        xls.to_excel("results_log_AlphaZero.xls")
    return soc

if __name__ == "__main__":
    eval_train = deque(maxlen=MEMORY_LENGTH)
    forecast_train = deque(maxlen=MEMORY_LENGTH)
    forecast_network = None
    evaluator_network = evaluator_net()
    env = ems(EPISODE_LENGTH)
    # forecast_network.load_weights("./networks/forecast_weights.h5")
    # evaluator_network.load_weights("./networks/evaluator_weights.h5")
    for episode in tqdm(range(EPISODES)):
        # eval_train = []
        # forecast_train = []
        state = env.reset()
        cum_reward = 0
        done = False
        while not done:
            action = np.argmax(evaluator_network.predict(np.expand_dims(state[4:], axis=0))[0])
            if np.random.random() > 0.8:
                action = np.random.randint(3)
            next_state, reward, done, info = env.step(action)
            eval_train.append([state, reward, action])
            forecast_train.append([state, next_state])
            state = next_state
            cum_reward += reward
        tqdm.write(f"Current episode resulted in {cum_reward} points of reward.")
        eval_x_train, eval_y_train = prepare_eval_train(np.array(eval_train), evaluator_network)
        # forec_x_train, forec_y_train = prepare_forecast_train(forecast_train)
        evaluator_network.fit(eval_x_train, eval_y_train, epochs=10, verbose=0)
        # forecast_network.train_on_batch(forec_x_train, forec_y_train)
        if not (episode+1) % PRINT_EVERY_X_ITER:
            soc = plot_test(evaluator_network, forecast_network)
