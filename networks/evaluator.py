from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD


def evaluator_net(obs=6, actions=3):
    inp = Input((obs,))
    x = Dense(24, activation='relu')(inp)
    x = Dense(24, activation='relu')(x)
    x = Dense(24, activation='relu')(x)
    out0 = Dense(actions, activation='softmax', name="actor_head")(x)
    out1 = Dense(1, activation="tanh", name="value_head")(x)

    m = Model(inp, [out0, out1])
    m.compile(optimizer = SGD(0.0001, momentum = 0.9), loss={"actor_head": "categorical_crossentropy", "value_head": "MSE"})
    return m