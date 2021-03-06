import numpy as np
import matplotlib.pyplot as plt


def euclidean_dist(a, b):
    return np.sqrt(np.sum(np.square(a - b)))

def reward_function(a, mu, f, epsilon=10):
    r = np.sum([f[i] * np.e ** (-(epsilon * (euclidean_dist(a, mu[i]))) ** 2) \
                for i in range(len(mu))])
    return r

def evaluate_outputs(input_state, outputs, num_samples=32, eta=0.05):
    outcomes = []
    rewards = []
    f = np.random.uniform(0, 4, 16)

    for i in range(num_samples):
        ind = i % len(outputs)
        pert = np.random.normal(0, 2 * [eta])
        outcome = outputs[ind] + pert
        reward = reward_function(outcome, input_state, f)
        outcomes.append(outcome)
        rewards.append(reward)

    return outcomes, rewards

def batch_evaluate(batch_inputs, batch_outputs):
    batch_outcomes = []
    batch_rewards = []

    for ind in range(len(batch_inputs)):
        outcomes, rewards = evaluate_outputs(batch_inputs[ind], batch_outputs[ind], eta=0.05)
        batch_outcomes.append(outcomes)
        batch_rewards.append(rewards)

    return np.array(batch_outcomes), np.array(batch_rewards)

def calc_regret(input_state, outputs, num):
    f = np.random.uniform(0, 4, 16)
    eta = 0.05

    bases = [.125, .375, .625, .875]

    uni_actions = []

    for i in bases:
        for j in bases:
            uni_actions.append([i, j])

    pred_a = one_ply(input_state, outputs, f)
    true_a = input_state[np.argmax(f)]
    uni_a = uni_actions[np.argmax(f)]
    peaks = np.random.uniform(0, 1, [16, 2])
    wrong_a = peaks[np.argmax(f)]


    pred_v = 0
    true_v = 0
    uni_v = 0
    wrong_v = 0

    for i in range(1000):
        pert = np.random.normal(0, 2 * [eta])
        outcome = pred_a + pert
        pred_v += reward_function(outcome, input_state, f)

    for i in range(1000):
        pert = np.random.normal(0, 2 * [eta])
        outcome = true_a + pert
        true_v += reward_function(outcome, input_state, f)

    for i in range(1000):
        pert = np.random.normal(0, 2 * [eta])
        outcome = wrong_a + pert
        wrong_v += reward_function(outcome, peaks, f)

    for i in range(1000):
        pert = np.random.normal(0, 2 * [eta])
        outcome = uni_a + pert
        uni_v += reward_function(outcome, input_state, f)

    pred_v /= 1000
    true_v /= 1000
    uni_v /= 1000
    wrong_v /= 1000

    if true_v - pred_v < 0:
        plot_reward_space(input_state, f, num, outputs)

    return true_v - pred_v, true_v - uni_v, wrong_v - pred_v

def batch_regret(batch_inputs, batch_outputs, i):
    regret = np.zeros(3)

    for ind in range(len(batch_inputs)):
        regret += calc_regret(batch_inputs[ind], batch_outputs[ind], ind + i)

    return regret / len(batch_inputs)

def one_ply(input_state, actions, f, num_samples=10):
    actions_stats = np.zeros(len(actions))
    eta = 0.05

    total_samples = len(actions) * num_samples

    for i in range(total_samples):
        ind = i % len(actions)
        pert = np.random.normal(0, 2 * [eta])
        outcome = actions[ind] + pert
        reward = reward_function(outcome, input_state, f)
        actions_stats[ind] += reward

    actions_stats = actions_stats / num_samples

    return actions[np.argmax(actions_stats)]

def plot_reward_space(mu, f, num, actions=None):
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

    fig = plt.figure()
    r_space = np.array(r_space)
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    plt.xticks(np.arange(0, len(x), 10), np.arange(0, 11) / 10)
    plt.yticks(np.arange(0, len(x), 10), np.arange(0, 11) / 10)
    plt.imshow(r_space, cmap='gray', vmin=0, vmax=4)
    plt.colorbar()
    plt.contour(r_space, alpha=0.6, linewidths=1, colors='black')
    a = mu[:, 1]
    b = mu[:, 0]
    plt.scatter(a * 100, b * 100, color='red', s=20, alpha=1, marker='x',
                zorder=2)

    if np.size(actions):
        a = actions[::, 1]
        b = actions[::, 0]
        plt.scatter(a * 100, b * 100, color='blue', s=20, alpha=1, marker='x',
                    zorder=3)

    plt.grid(alpha=0.75, linestyle='dashed')
    fname = 'fig_{}'.format(int(num))
    plt.savefig(fname=fname)
    plt.close()


# mu = np.random.uniform(0, 1, (20, 2))
# f = np.random.uniform(0, 4, 20)
# off = 0#np.random.choice(range(0, 5))
# mu = [[.25, .25], [0.5, 0.5], [.75, .75]]
# f = [.5, 2, 1.]
# a = [[.25, .25], [.5, .5], [.75, .75]]

# stats = one_ply(mu, a, f)
# print(stats)

# plot_reward_space(mu, f, off)


# batch_inputs = [[[0.25, 0.30],
#                  [0.45, 0.57],
#                  [0.77, 0.23]],
#                 [[0.78, 0.51],
#                  [0.81, 0.93],
#                  [0.67, 0.17]]]

# batch_outputs = [[[0.22, 0.30],
#                   [0.41, 0.37],
#                   [0.47, 0.25]],
#                  [[0.68, 0.90],
#                   [0.89, 0.13],
#                   [0.37, 0.27]]]

# batch_outcomes, batch_rewards = batch_evaluate(batch_inputs, batch_outputs)
# print(batch_outcomes)
# print(batch_rewards)
