import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ems_env import ems
from mcts import uct_search, StateNode
from networks import evaluator, forecaster

EPISODE_LENGTH = 480
SEARCH_DEPTH = 50
high_score = 453


def prepare_eval_train(eval_train, evaluator_network):
    eval_train = np.array(eval_train)
    states = np.stack(eval_train[:, 0])[:, 4:]
    actions = np.argmax(np.stack(eval_train[:, 2]), axis=1)
    rewards = np.stack(eval_train[:, 1])
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


def evaluate_current_iteration(LOGFILE=False, PLOT=False):
    forecast_network = forecaster.forecast_net()
    evaluator_network = evaluator.evaluator_net()
    forecast_network.load_weights("./networks/forecast_weights.h5")
    evaluator_network.load_weights("./networks/evaluator_weights.h5")
    test_env = ems(960)
    state = test_env.reset()
    test_env.time = 20000
    log, soc = [], []
    cum_r = 0
    for i in range(960):
        action, root = uct_search(StateNode(state, test_env.time), SEARCH_DEPTH, evaluator=evaluator_network, forecaster=forecast_network, use_dirichlet=False)
        state, r, done, _ = test_env.step(action)
        log.append([action, state[4], state[5], r])
        soc.append(state[4])
        cum_r += r
    if cum_r > high_score:
        print(f"\n\n--=== New highscore achieved: {cum_r}! ===--\n\n")
        forecast_network.save_weights("./networks/best_forecast_weights.h5")
        evaluator_network.save_weights("./networks/best_evaluator_weights.h5")
        high_score = cum_r
    if PLOT:
        pd.DataFrame(soc).plot()
        plt.show(block=True)
        plt.close()
    if LOGFILE:
        xls = pd.DataFrame(log)
        xls.to_excel("results_log_AlphaZero.xls")
    return soc


def create_training_samples():
    eval_train, forecast_train = [], []
    forecast_network = forecaster.forecast_net()
    evaluator_network = evaluator.evaluator_net()
    env = ems(EPISODE_LENGTH)
    forecast_network.load_weights("./networks/forecast_weights.h5")
    evaluator_network.load_weights("./networks/evaluator_weights.h5")
    state = env.reset()
    done = False
    while not done:
        action, uct_node = uct_search(StateNode(state, env.time), SEARCH_DEPTH,
                                      evaluator=evaluator_network,
                                      forecaster=forecast_network)
        next_state, reward, done, info = env.step(action)
        eval_train.append([state, reward, uct_node.child_number_visits])
        forecast_train.append([state, next_state])
        state = next_state
    return [eval_train, forecast_train]


def training_step(eval_train, forecast_train):
    loss = "loss"
    tack = time.time()
    forecast_network = forecaster.forecast_net()
    evaluator_network = evaluator.evaluator_net()
    forecast_network.load_weights("./networks/forecast_weights.h5")
    evaluator_network.load_weights("./networks/evaluator_weights.h5")
    eval_x_train, eval_y_train = prepare_eval_train(eval_train, evaluator_network)
    forecast_x_train, forecast_y_train = prepare_forecast_train(forecast_train)
    eval_hist = evaluator_network.fit(eval_x_train, eval_y_train, epochs=250, verbose=0)
    forecast_hist = forecast_network.fit(forecast_x_train, forecast_y_train, epochs=10, verbose=0)
    forecast_network.save_weights("./networks/forecast_weights.h5")
    evaluator_network.save_weights("./networks/evaluator_weights.h5")
    tick = time.time()

    print(f"Training phase took {int(tick-tack)} seconds."
          f"\nEvaluation Loss: {np.round(eval_hist.history[loss][-1], 4)}"
          f"\nForecast Loss: {np.round(forecast_hist.history[loss][-1], 4)}")
    return eval_hist, forecast_hist
