import logging
import os
import time
from multiprocessing import cpu_count, Pool

import numpy as np

from mcts import calc_time_waves
from networks import forecaster
from training import create_training_samples, training_phase, evaluate_current_iteration

os.system("taskset -p 0xff %d" % os.getpid())
os.environ['KMP_WARNINGS'] = 'off'
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

EPISODES = 50
PRINT_EVERY_X_ITER = 5
HIGH_SCORE = 564


def prepare_forecast_timeseries(forecast_network):
    timeseries = list(map(calc_time_waves, range(35040)))
    forecast = forecast_network.predict(np.array(timeseries))
    return np.array(forecast)


def results(result):
    global eval_train, forecast_train
    eval_train += result[0]
    forecast_train += result[1]


def data_gathering_phase(p):
    print(f"\n_____________________________________________"
          f"\nStarting episode {episode+1} of {EPISODES}")
    forecast_network = forecaster.forecast_net()
    try:
        forecast_network.load_weights("./networks/forecast_weights.h5")
    except:
        print("No weights loaded")
    forecast_timeseries = prepare_forecast_timeseries(forecast_network)
    tack = time.time()
    workers = [p.apply_async(create_training_samples, args=(forecast_timeseries, ),
                             callback=results) for i in range(cpu_count())]
    [w.wait() for w in workers]
    tick = time.time()
    print(f"Data gathering phase took {int(tick-tack)} seconds.")
    return forecast_timeseries


def evaluation_phase(high_score, eval_result, forecast_timeseries):
    if eval_result == "":
        print("Evaluation loop initiated...")
        eval_result = pool.apply_async(evaluate_current_iteration, args=(high_score, forecast_timeseries))
    elif eval_result.successful():
        print("Evaluation loop initiated...")
        high_score = eval_result.get()
        eval_result = pool.apply_async(evaluate_current_iteration, args=(high_score, forecast_timeseries))
    return high_score, eval_result


if __name__ == '__main__':
    os.environ["OPENBLAS_MAIN_FREE"] = "1"
    pool = Pool(cpu_count())
    high_score = HIGH_SCORE
    eval_result = ""
    for episode in range(EPISODES):
        eval_train, forecast_train = [], []
        forecast_timeseries = data_gathering_phase(pool)
        eval_hist, forecast_hist = training_phase(eval_train, forecast_train)
        if not (episode+1) % PRINT_EVERY_X_ITER:
            high_score, eval_result = evaluation_phase(high_score, eval_result, forecast_timeseries)

