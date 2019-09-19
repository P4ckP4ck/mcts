decisions = []
env = VPPirgendwas()
for episode in range(25):
    state = env.reset()
    while not done:
        action = UCT_search(state)
        state, r, done = env.step(action)
        decisions.append([state, r, UCT_search.child_number_visits])
    reward = get_reward(decisions) #needs to be the
    x, y = prepare_trainingset(decisions, reward)
    NN.train_on_batch(x, y)
