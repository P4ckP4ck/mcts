import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from ems_env import ems
from mcts import UCT_search, StateNode
from networks import evaluator

evaluator_network = evaluator.evaluator_net()

test_env = ems(96)
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
