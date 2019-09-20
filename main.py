from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ems_env import ems
from mcts import UCT_search, StateNode
from networks import evaluator, forecaster

EPISODES = 100
PRINT_EVERY_X_ITER = 10
EPISODE_LENGTH = 480
SEARCH_DEPTH = 40
MEMORY_LENGTH = 10000

def prepare_eval_train(eval_train, evaluator_network):
    eval_train = np.array(eval_train)
    states = np.stack(eval_train[:,0])[:,4:]
    actions = np.argmax(np.stack(eval_train[:,2]), axis=1)
    rewards = np.stack(eval_train[:,1])
    x = states
    # action_priors = pd.DataFrame(np.vstack(eval_df[2])/eval_df[2].sum()) #soft update version
    index = range(x.shape[0])
    action_priors = np.zeros((x.shape[0], 3))
    action_priors[index, actions] = 1
    values = evaluator_network.predict(x)[1]
    values[index, actions] = rewards
    return x, [action_priors, values]

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
        action, root = UCT_search(StateNode(state, env.time), SEARCH_DEPTH, evaluator=evaluator_network, forecaster=forecast_network, use_dirichlet=False)
        state, r, done, _ = test_env.step(action)
        log.append([action, state[4], state[5], r])
        soc.append(state[4])
        cum_r += r
    tqdm.write(f" Current weights achieve a score of {cum_r}")
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
    forecast_network = forecaster.forecast_net()
    evaluator_network = evaluator.evaluator_net()
    env = ems(EPISODE_LENGTH)
    forecast_network.load_weights("./networks/forecast_weights.h5")
    evaluator_network.load_weights("./networks/evaluator_weights.h5")
    for episode in tqdm(range(EPISODES)):
        state = env.reset()
        cum_reward = 0
        done = False
        while not done:
            action, UCT_node = UCT_search(StateNode(state, env.time), SEARCH_DEPTH, evaluator=evaluator_network, forecaster=forecast_network)
            next_state, reward, done, info = env.step(action)
            eval_train.append([state, reward, UCT_node.child_number_visits])
            forecast_train.append([state, next_state])
            state = next_state
            cum_reward += reward
        tqdm.write(f"Current episode resulted in {cum_reward} points of reward.")
        eval_x_train, eval_y_train = prepare_eval_train(eval_train, evaluator_network)
        forec_x_train, forec_y_train = prepare_forecast_train(forecast_train)
        evaluator_network.fit(eval_x_train, eval_y_train, epochs=15, verbose=0)
        forecast_network.train_on_batch(forec_x_train, forec_y_train)
        if not (episode+1) % PRINT_EVERY_X_ITER:
            soc = plot_test(evaluator_network, forecast_network)
    forecast_network.save_weights("./networks/forecast_weights.h5")
    evaluator_network.save_weights("./networks/evaluator_weights.h5")
