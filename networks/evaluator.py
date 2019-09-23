import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD


def evaluator_net(obs=3, actions=3):
    inp = Input((obs,))
    x = Dense(24, activation='relu')(inp)
    # x = Dense(24, activation='relu')(x)
    x_val = Dense(24, activation='relu')(x)
    x_act = Dense(24, activation='relu')(x)
    out0 = Dense(actions, activation='softmax', name="actor_head")(x_act)
    out1 = Dense(actions, activation="linear", name="value_head")(x_val)

    m = Model(inp, [out0, out1])
    m.compile(optimizer=SGD(0.0001, momentum=0.9),
              loss={"actor_head": "categorical_crossentropy", "value_head": _huber_loss})
    return m


def _huber_loss(y_true, y_pred, clip_delta=1.0):
    # source: keon.io #
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta

    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

    return K.mean(tf.where(cond, squared_loss, quadratic_loss))