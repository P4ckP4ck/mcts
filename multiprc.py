import logging
import os
import time
from multiprocessing import cpu_count, Pool

from training import create_training_samples, training_step, evaluate_current_iteration

os.system("taskset -p 0xff %d" % os.getpid())
os.environ['KMP_WARNINGS'] = 'off'
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

EPISODES = 50
PRINT_EVERY_X_ITER = 5


def results(result):
    global eval_train, forecast_train
    eval_train += result[0]
    forecast_train += result[1]


def data_gathering_task(pool):
    print(f"\n_____________________________________________"
          f"\nStarting episode {episode+1} of {EPISODES}")
    tack = time.time()
    workers = [pool.apply_async(create_training_samples, callback=results) for i in range(cpu_count())]
    [w.wait() for w in workers]
    tick = time.time()
    print(f"Data gathering phase took {int(tick-tack)} seconds.")

if __name__ == '__main__':
    os.environ["OPENBLAS_MAIN_FREE"] = "1"
    pool = Pool(cpu_count())
    eval_result = None
    for episode in range(EPISODES):
        eval_train, forecast_train = [], []
        data_gathering_task(pool)
        eval_hist, forecast_hist = training_step(eval_train, forecast_train)
        if not (episode+1) % PRINT_EVERY_X_ITER:
            if eval_result == None or eval_result.ready():
                print("Evaluation loop initiated...")
                eval_result = pool.apply_async(evaluate_current_iteration)
