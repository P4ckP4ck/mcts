import logging
import os
import time
from multiprocessing import cpu_count, Pool

from training import create_training_samples, training_phase, evaluate_current_iteration

os.system("taskset -p 0xff %d" % os.getpid())
os.environ['KMP_WARNINGS'] = 'off'
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

EPISODES = 50
PRINT_EVERY_X_ITER = 5
HIGH_SCORE = 564


def results(result):
    global eval_train, forecast_train
    eval_train += result[0]
    forecast_train += result[1]


def data_gathering_phase(p):
    print(f"\n_____________________________________________"
          f"\nStarting episode {episode+1} of {EPISODES}")
    tack = time.time()
    workers = [p.apply_async(create_training_samples, callback=results) for i in range(cpu_count())]
    [w.wait() for w in workers]
    tick = time.time()
    print(f"Data gathering phase took {int(tick-tack)} seconds.")


def evaluation_phase(high_score, eval_result):
    if eval_result == "":
        print("Evaluation loop initiated...")
        eval_result = pool.apply_async(evaluate_current_iteration, args=(high_score, ))
    elif eval_result.successful():
        print("Evaluation loop initiated...")
        high_score = eval_result.get()
        eval_result = pool.apply_async(evaluate_current_iteration, args=(high_score, ))
    return high_score, eval_result


if __name__ == '__main__':
    os.environ["OPENBLAS_MAIN_FREE"] = "1"
    pool = Pool(cpu_count())
    high_score = HIGH_SCORE
    eval_result = ""
    for episode in range(EPISODES):
        eval_train, forecast_train = [], []
        data_gathering_phase(pool)
        eval_hist, forecast_hist = training_phase(eval_train, forecast_train)
        if not (episode+1) % PRINT_EVERY_X_ITER:
            high_score, eval_result = evaluation_phase(high_score, eval_result)

