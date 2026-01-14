import numpy as np

actions = ["success", "left", "right", "other"]


nprng = np.random.default_rng(6)


for x in range(100):
    next_action = nprng.choice(actions, p=[0.25, 0.25, 0.25, 0.25])
    print(next_action)