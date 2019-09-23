import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from complex_ems import ComplexEMS as ems
from mcts import uct_search, StateNode
from networks import evaluator, forecaster

EPISODE_LENGTH = 480
SEARCH_DEPTH = 50


def prepare_eval_train(eval_train, evaluator_network):
    eval_train = np.array(eval_train)
    states = np.stack(eval_train[:, 0])
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
    x = states[states.columns[0:6]]
    return x, y


def evaluate_current_iteration(high_score, forecast, LOGFILE=False, PLOT=False):
    evaluator_network = evaluator.evaluator_net()
    evaluator_network.load_weights("./networks/evaluator_weights.h5")
    test_env = ems(960)
    state = test_env.reset()
    test_env.time = 20000
    log, soc = [], []
    cum_r = 0
    for i in range(960):
        action, root = uct_search(StateNode(state, test_env), SEARCH_DEPTH, forecast, evaluator_network=evaluator_network, use_dirichlet=False)
        state, r, done, _ = test_env.step(action)
        log.append([action, state[0], state[1], state[2], r])
        soc.append(state[0])
        cum_r += r

    if cum_r > high_score:
        print(f"\n\n--=== New highscore achieved: {cum_r}! ===--\n\n")
        # forecast_network.save_weights("./networks/best_forecast_weights.h5")
        evaluator_network.save_weights("./networks/best_evaluator_weights.h5")
        high_score = cum_r
    else:
        print(f"No new highscore, current performance: {cum_r}!")

    if PLOT:
        pd.DataFrame(soc).plot()
        plt.show(block=True)
        plt.close()

    if LOGFILE:
        xls = pd.DataFrame(log)
        xls.to_excel("results_log_AlphaZero.xls")
    return high_score


def create_training_samples(forecast_timeseries):
    eval_train, forecast_train = [], []
    evaluator_network = evaluator.evaluator_net()
    try:
        evaluator_network.load_weights("./networks/evaluator_weights.h5")
    except:
        pass
    env = ems(EPISODE_LENGTH)
    state = env.reset()
    done = False
    while not done:
        action, uct_node = uct_search(StateNode(state, env), SEARCH_DEPTH,
                                      forecast_timeseries, evaluator_network=evaluator_network)
        next_state, reward, done, info = env.step(action)
        eval_train.append([state, reward, uct_node.child_number_visits])
        forecast_train.append([state, next_state])
        state = next_state
    return [eval_train, forecast_train]


def training_phase(eval_train):
    loss = "loss"
    tack = time.time()
    forecast_network = forecaster.forecast_net()
    evaluator_network = evaluator.evaluator_net()
    try:
        forecast_network.load_weights("./networks/forecast_weights.h5")
    except:
        print("No forecast weights loaded")
    try:
        evaluator_network.load_weights("./networks/evaluator_weights.h5")
    except:
        pass
    eval_x_train, eval_y_train = prepare_eval_train(eval_train, evaluator_network)
    eval_hist = evaluator_network.fit(eval_x_train, eval_y_train, epochs=250, verbose=0)
    evaluator_network.save_weights("./networks/evaluator_weights.h5")
    tick = time.time()

    print(f"Training phase took {int(tick-tack)} seconds."
          f"\nEvaluation Loss: {np.round(eval_hist.history[loss][-1], 4)}")
    return eval_hist

if __name__ == "__main__":
    a = create_training_samples([])