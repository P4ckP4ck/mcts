from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import SGD


def evaluator_net(obs=2, actions=3):
    inp = Input((obs,))
    x = Dense(24, activation='relu')(inp)
    # x = Dense(24, activation='relu')(x)
    x_val = Dense(24, activation='relu')(x)
    x_act = Dense(24, activation='relu')(x)
    out0 = Dense(actions, activation='softmax', name="actor_head")(x_act)
    out1 = Dense(1, activation="tanh", name="value_head")(x_val)

    m = Model(inp, [out0, out1])
    m.compile(optimizer = SGD(0.0001, momentum = 0.9), loss={"actor_head": "categorical_crossentropy", "value_head": "MSE"})
    return m