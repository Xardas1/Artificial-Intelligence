from yarl.environments import make_environment
from yarl.algorithms import make_algorithm

env = make_environment('car-parking')
alg = make_algorithm('proximal-policy-optimization')

alg.random_init(len(env.ss))
for epoch in range(100):
    batches = []
    for n in range(100):
        samples = []
        s = env.random_state()
        end_state = False
        while end_state:
            a = alg.get_actions(s)
            s1,end_state,reward = env.step(s,a)
            samples.append((s,a))
            s = s1
        samples = alg.apply_reward(samples,reward)
        batches.append(samples)
    alg = alg.train( batches )

print(env.get_info())