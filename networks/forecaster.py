import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD


atoms = 11


def forecast_net(ATOMS=atoms):
    inp = Input((6,))
    x = Dense(24, activation='relu')(inp)
    x = Dense(24, activation='relu')(x)
    x = Dense(24, activation='relu')(x)
    out = (Dense(ATOMS, activation='softmax')(x))
    m = Model(inp, out)
    m.compile(optimizer=SGD(0.0001, momentum=0.9), loss="categorical_crossentropy")
    return m


def prepare_forecast_set(y, ATOMS=atoms):
    r_max = 1
    r_min = -1
    delta_r = (r_max - r_min) / float(ATOMS - 1)
    z = [r_min + i * delta_r for i in range(ATOMS)]
    m_probs = np.zeros((y.shape[0], ATOMS))
    Tz = np.clip(y, r_min, r_max)
    bj = (Tz - r_min) / delta_r
    m_l, m_u = np.floor(bj).astype("int32"), np.ceil(bj).astype("int32")
    index = np.arange(0, y.shape[0]).astype("int32")
    where_equal = np.equal(m_l, m_u)
    if any(where_equal):
        m_probs[index[where_equal], m_l[where_equal]] = 1
    if any(~where_equal):
        m_probs[index[~where_equal], m_l[~where_equal]] = (m_u - bj)[~where_equal]
        m_probs[index[~where_equal], m_u[~where_equal]] = (bj - m_l)[~where_equal]
    return m_probs
