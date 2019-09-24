import gc
import logging
import os
import time
from collections import deque

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
HIGH_SCORE = 0


def prepare_forecast_timeseries(forecast_network):
    timeseries = list(map(calc_time_waves, range(35040)))
    forecast = forecast_network.predict(np.array(timeseries))
    return np.array(forecast)


def results(result):
    global eval_train, forecast_train
    eval_train += result[0]
    forecast_train += result[1]


def data_gathering_phase():
    print(f"\n_____________________________________________"
          f"\nStarting episode {episode+1} of {EPISODES}")
    forecast_network = forecaster.forecast_net()
    try:
        forecast_network.load_weights("./networks/forecast_weights.h5")
    except:
        print("No weights loaded")
    forecast_timeseries = []# prepare_forecast_timeseries(forecast_network)
    tack = time.time()
    training_samples = create_training_samples(forecast_timeseries)
    tick = time.time()
    print(f"Data gathering phase took {int(tick-tack)} seconds.")
    return training_samples[0], forecast_timeseries


def evaluation_phase(high_score, forecast_timeseries):
    high_score = evaluate_current_iteration(high_score, forecast_timeseries)
    return high_score


if __name__ == '__main__':
    os.environ["OPENBLAS_MAIN_FREE"] = "1"
    high_score = HIGH_SCORE
    eval_result = ""
    eval_train_deque= deque(maxlen=5000)
    import ctypes
    for episode in range(EPISODES):
        eval_train, forecast_train = [], []
        eval_train, forecast_timeseries = data_gathering_phase()
        eval_train_deque += eval_train
        eval_hist = training_phase(eval_train_deque)
        if not (episode+1) % PRINT_EVERY_X_ITER:
            high_score = evaluation_phase(high_score, forecast_timeseries)
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
