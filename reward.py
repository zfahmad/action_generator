import numpy as np


def evaluate(batch_actions, batch_peaks):
    batch_outcomes = []
    batch_values = []

    for actions, peaks in zip(batch_actions, batch_peaks):
        outcomes, values = sample(peaks, actions, 30)
        batch_outcomes.append(outcomes)
        batch_values.append(values)

    return batch_outcomes, batch_values


def sample(peaks, actions, num_samples):
    eta = 0.01
    f = np.random.uniform(.5, 2, [3])

    a_realized = []
    v_realized = []

    for i in range(num_samples):
        ind = i % len(actions)
        a = actions[ind]
        a_ = a + np.random.normal(0, eta)
        r = foo(a_, peaks, f)

        a_realized.append(a_)
        v_realized.append([r])

    return np.array(a_realized), np.array(v_realized)


def foo(a, mu, f, epsilon=10):
    r = np.sum([f[i] * np.e ** (-(epsilon * (a - mu[i])) ** 2) \
                for i in range(len(mu))])
    return r
