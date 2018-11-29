import numpy as np
import matplotlib.pyplot as plt


def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def reward_function(a, mu, f, epsilon=10):
    r = np.sum([f[i] * np.e ** (-(epsilon * (euclidean_dist(a, mu[i]))) ** 2) \
                for i in range(len(mu))])
    return r

def evaluate_outputs(input, outputs, num_samples=6, eta=0.05):
    outcomes = []
    rewards = []
    f = np.random.uniform(-4, 4, 15)

    for i in range(num_samples):
        ind = i % len(outputs)
        pert = np.random.normal(0, 2 * [eta])
        outcome = outputs[ind] + pert
        reward = reward_function(outcome, input, f)
        outcomes.append(outcome)
        rewards.append(reward)

    return outcomes, rewards

def batch_evaluate(batch_inputs, batch_outputs):
    batch_outcomes = []
    batch_rewards = []

    for ind in range(len(batch_inputs)):
        outcomes, rewards = evaluate_outputs(batch_inputs[ind], batch_outputs[ind])
        batch_outcomes.append(outcomes)
        batch_rewards.append(rewards)

    return np.array(batch_outcomes), np.array(batch_rewards)

def plot_reward_space(mu, f):
    x = np.linspace(0, 1, 101)
    y = np.linspace(0, 1, 101)

    r_space = []
    domain = np.meshgrid(x, y)

    for i in x:
        row = []
        for j in y:
            a = np.array([i, j])
            row.append(reward_function(a, mu, f))

        r_space.append(row)

    r_space.reverse()
    r_space = np.array(r_space)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xticks(np.arange(0, len(x), 10), np.arange(0, 11) / 10)
    plt.yticks(np.arange(0, len(x), 10), np.arange(10, -1, -1) / 10)
    plt.imshow(r_space, cmap='OrRd', vmin=-4, vmax=4)
    plt.colorbar()
    plt.contour(r_space, levels=12, alpha=0.6, linewidths=1, cmap='gray')
    a = mu[::, 1]
    b = mu[::, 0]
    plt.scatter(a * 100, 100 - b * 100, color='blue', s=20, alpha=1, marker='x',
                zorder=2)
    plt.show()


mu = np.random.uniform(0, 1, (15, 2))
f = np.random.uniform(-4, 4, 15)

plot_reward_space(mu, f)

# batch_inputs = [[[0.25, 0.30],
#                  [0.45, 0.57],
#                  [0.77, 0.23]],
#                 [[0.78, 0.51],
#                  [0.81, 0.93],
#                  [0.67, 0.17]]]
#
# batch_outputs = [[[0.22, 0.30],
#                   [0.41, 0.37],
#                   [0.47, 0.25]],
#                  [[0.68, 0.90],
#                   [0.89, 0.13],
#                   [0.37, 0.27]]]
#
# batch_outcomes, batch_rewards = batch_evaluate(batch_inputs, batch_outputs)
# print(batch_outcomes)
# print(batch_rewards)
