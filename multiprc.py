
import os
import time
from multiprocessing import cpu_count, Process

from tqdm import tqdm

from training import create_training_samples

os.system("taskset -p 0xff %d" % os.getpid())
def results(result):
    global et, ft
    et.append(result[0])
    ft.append(result[1])


if __name__ == '__main__':
    os.environ["OPENBLAS_MAIN_FREE"] = "1"
    # pool = Pool(cpu_count())
    for episode in tqdm(range(1)):
        tack = time.time()
        # ft, et = [], []
        workers = [Process(target=create_training_samples) for i in range(cpu_count())]
        [w.start() for w in workers]
        [w.join() for w in workers]
        # for w in workers:
        #     w.wait()
        tick = time.time()
        # et = np.vstack(et)
        # ft = np.vstack(ft)
        tqdm.write(f"Data gathering phase took {int(tick-tack)} seconds.")
        # tack = time.time()
        # training_step(et, ft)
        # tick = time.time()
        # tqdm.write(f"Training phase took {int(tick-tack)} seconds.")
        # soc = plot_test()



