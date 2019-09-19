import math

import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD
from tqdm import tqdm

###!IMPLEMENT FORECASTS###
atoms = 51

def network():
    inp = Input((4,))
    x = Dense(12, activation='relu')(inp)
    for _ in range(2 - 1):
        x = Dense(12, activation='relu')(x)
    out = (Dense(atoms, activation='softmax')(x))
    M = Model(inp, out)
    M.compile(optimizer = SGD(0.0001, momentum = 0.9), loss = "categorical_crossentropy")
    return M

predictor = network()
pv = pd.read_csv("./Input_House/PV/Solar_Data-Random.csv", delimiter = ";")#, skiprows = 0)
lp = pd.read_csv("./Input_House/Base_Szenario/df_S_15min.csv", delimiter = ",")
day_sin = []
for i, rad in enumerate(data["Generation"]):
    sin_day = np.sin(2*np.pi*i/(24*4))
    cos_day = np.cos(2*np.pi*i/(24*4))
    sin_year = np.sin(2*np.pi*i/(24*4*365))
    cos_year = np.cos(2*np.pi*i/(24*4*365))
    day_sin.append([sin_day, cos_day, sin_year, cos_year])

df = pd.DataFrame(day_sin)
df.plot(subplots = True)#

r_max = 0.9
r_min = -0.1
delta_r = (r_max - r_min) / float(atoms - 1)
z = [r_min + i * delta_r for i in range(atoms)]


m_prob = np.zeros((35040, atoms))
for i, reward in enumerate(data["Generation"]):
    Tz = min(r_max, max(r_min, reward))
    bj = (Tz - r_min) / delta_r
    m_l, m_u = math.floor(bj), math.ceil(bj)
    m_prob[i][int(m_l)] += (m_u - bj)
    m_prob[i][int(m_u)] += (bj - m_l)
predictor.fit(df, m_prob, epochs = 15, verbose = 1)

for j in range(96):
    a = predictor.predict(df.iloc[(96*90+j):(96*90+j+1)])
    d = pd.DataFrame(a)
    d.T.plot.bar(subplots = True, title = f"Time: {int(j/4)%24}:00, Day: {int(j/(4*24))}", ylim = (0,0.9))#
    plt.savefig(f'./plots/pv{j}.png')
    plt.close()

images = []
for filename in tqdm(range(96)):
    images.append(imageio.imread(f"./plots/pv{filename}.png"))
imageio.mimsave('pv.gif', images, duration=0.06)