import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from ems_env import ems
from mcts import UCT_search, StateNode
from networks import evaluator, forecaster

EPISODES = 25
PRINT_EVERY_X_ITER = 5
EPISODE_LENGTH = 480
SEARCH_DEPTH = 15
MEMORY_LENGTH = 20000

def prepare_eval_train(eval_train):
    eval_df = pd.DataFrame(eval_train)
    states = pd.DataFrame(np.vstack(eval_df[0]))
    x = states[states.columns[4:]]
    # action_priors = pd.DataFrame(np.vstack(eval_df[2])/eval_df[2].sum())
    index = range(eval_df.shape[0])
    action_priors = np.zeros((eval_df.shape[0], 3))
    action_priors[index, np.argmax(np.vstack(eval_df[2]), axis=1)] = 1
    values = pd.DataFrame(np.vstack(eval_df[1]))
    return x, [action_priors, values]

def prepare_forecast_train(forecast_train):
    forecast_df = pd.DataFrame(forecast_train)
    states = pd.DataFrame(np.vstack(forecast_df[0]))
    next_states = pd.DataFrame(np.vstack(forecast_df[1]))
    y = forecaster.prepare_forecast_set(next_states[5])
    x = states[states.columns[0:4]]
    return x, y

def plot_test(evaluator_network, forecast_network, LOGFILE=True):
    test_env = ems(EPISODE_LENGTH)
    state = test_env.reset()
    test_env.time = 20000
    log, soc = [], []
    cum_r = 0
    for i in range(960):
        action, root = UCT_search(StateNode(state, env.time), SEARCH_DEPTH, evaluator=evaluator_network, forecaster=forecast_network, use_dirichlet=False)
        state, r, done, _ = test_env.step(action)
        log.append([action, state[0], state[1], state[2], r])
        soc.append(state[4])
        cum_r += r
    tqdm.write(f" Current weights achieve a score of {cum_r}")
    # if cum_r > self.high_score and self.SAVE_HIGHSCORE:
    #     self.high_score = cum_r
    #     self.actor_target.save_weights(f"high_score_weights_{cum_r}.h5")
    pd.DataFrame(soc).plot()
    plt.show(block=True)
    plt.close()
    if LOGFILE:
        xls = pd.DataFrame(log)
        xls.to_excel("results_log_AlphaZero.xls")
    return soc

if __name__ == "__main__":
    # eval_train = deque(maxlen=MEMORY_LENGTH)
    # forecast_train = deque(maxlen=MEMORY_LENGTH)
    forecast_network = forecaster.forecast_net()
    evaluator_network = evaluator.evaluator_net()
    env = ems(EPISODE_LENGTH)
    forecast_network.load_weights("./networks/forecast_weights.h5")
    evaluator_network.load_weights("./networks/evaluator_weights.h5")
    for episode in tqdm(range(EPISODES)):
        eval_train = []
        forecast_train = []
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
        eval_x_train, eval_y_train = prepare_eval_train(eval_train)
        forec_x_train, forec_y_train = prepare_forecast_train(forecast_train)
        evaluator_network.train_on_batch(eval_x_train, eval_y_train)
        forecast_network.train_on_batch(forec_x_train, forec_y_train)
        if not (episode+1) % PRINT_EVERY_X_ITER:
            soc = plot_test(evaluator_network, forecast_network)
    forecast_network.save_weights("./networks/forecast_weights.h5")
    evaluator_network.save_weights("./networks/evaluator_weights.h5")