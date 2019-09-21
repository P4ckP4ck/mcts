
from multiprocessing import Process, Manager, cpu_count, Pool
import time
from training import create_training_samples, training_step, plot_test
import numpy as np
import pandas as pd

# def create_training_samples(eval_train, forecast_train):
#     for i in range(5000):
#         eval_train.put(i)
#         forecast_train.put(np.sqrt(i))

def calc_training_set(num_processes = cpu_count()):
    eval_train = Manager().Queue()
    forecast_train = Manager().Queue()
    processes = []

    for w in range(num_processes):
        p = Process(target=create_training_samples(eval_train, forecast_train))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    return eval_train, forecast_train


if __name__ == '__main__':
    for episode in range(50):
        tack = time.time()
        pool = Pool(processes=cpu_count())
        m = Manager()
        eval_train = m.Queue()
        forecast_train = m.Queue()
        workers = pool.apply(create_training_samples, (eval_train, forecast_train))
        tick = time.time()
        print(tick-tack)
        #stop until done!!!
        ft, et = [], []
        while not forecast_train.empty():
            ft.append(forecast_train.get())
        while not eval_train.empty():
            et.append(eval_train.get())
        training_step(et, ft)

        soc = plot_test()



