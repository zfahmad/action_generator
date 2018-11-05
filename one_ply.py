import numpy as np

def test_net(batch_peaks, batch_actions):

    reg = []
    reg_uni = []

    for peaks, actions in zip(batch_peaks, batch_actions):

        mu_ = peaks
        actions = np.reshape(actions, [-1])
        f_ = np.random.uniform(0.5, 2.0, np.size(actions))

        a = kr_search(mu_, f_, actions, 25)
        reg.append(cal_regret(f_, mu_, mu_[np.argmax(f_)], a))

        uni_a = np.array([0.25, 0.5, 0.75])
        a = kr_search(mu_, f_, uni_a, 25)
        reg_uni.append(cal_regret(f_, mu_, mu_[np.argmax(f_)], a))

    return [np.mean(reg), np.mean(reg_uni)]


def one_ply(mu_, f_, actions, num_samples):
    a_stats = np.zeros(np.size(actions))
    a_counts = np.zeros(np.size(actions))


    for i in range(num_samples):
        ind = i % np.size(actions)
        r, a_ = test_func(mu_, f_, actions[ind])
        a_stats[ind] += r
        a_counts[ind] += 1

    a_stats = a_stats / a_counts

    return actions[np.argmax(a_stats)]


def cal_regret(f_, mu_, a, a_):
    regret = 0

    for i in range(1000):
        tru_r, _ = test_func(mu_, f_, a)
        obs_r, _ = test_func(mu_, f_, a_)
        regret += tru_r - obs_r

    return regret / 1000


def test_func(mu_, f_, a, eta=0.05):
    epsilon = 20
    a_ = a + np.random.normal(0, eta)
    r = np.sum([f_[i] * np.e ** (-(epsilon * (a_ - mu_[i])) ** 2) for i in range(len(mu_))])

    return r, a_


def gauss_kernel(u, sigma=0.05):
    return (1 / (sigma * np.sqrt(2 * np.pi)) * np.e ** (-0.5 * ((u / sigma) ** 2)))


def kde(a, b, K, h=1):
    return K((a - b) / h)


def kr_search(mu_, f_, actions, num_samples):
    tau = 0.9
    A = np.array([])
    A = np.append(A, actions)
    a_stats = np.ones(np.size(actions)) * np.inf
    a_vals = np.zeros(np.size(actions))
    a_counts = np.zeros(np.size(actions))

    for i in range(np.size(A)):
        a = actions[i]
        r, a_ = test_func(mu_, f_, a)

        a_vals[i] = ((a_vals[i] * a_counts[i]) + r) / (a_counts[i] + 1)
        a_counts[i] += 1

    a_stats = update_kr(A, a_vals, a_counts)

    for i in range(num_samples - 3):
        sel_ind = np.argmax(a_stats)
        a = A[sel_ind]
        r, a_ = test_func(mu_, f_, a)

        if kde(a, a_, gauss_kernel) > tau:
            A = np.append(A, a_)
            a_stats = np.append(a_stats, np.inf)
            a_vals = np.append(a_vals, r)
            a_counts = np.append(a_counts, 1)

        a_vals[sel_ind] = ((a_vals[sel_ind] * a_counts[sel_ind]) + r) / (a_counts[sel_ind] + 1)
        a_counts[sel_ind] += 1
        a_stats = update_kr(A, a_vals, a_counts)

    return A[np.argmax(a_vals)]


def update_kr(actions, a_vals, a_counts, C=1):

    stats = np.zeros(np.size(actions))
    e_a = np.zeros(np.size(actions))
    norm = np.zeros(np.size(actions))
    tot = 0

    for i in range(len(actions)):
        e_a_i = 0
        for j in range(len(actions)):
            e_a_i += kde(actions[i], actions[j], gauss_kernel) * a_vals[j] * a_counts[j]
            norm[i] = kde(actions[i], actions[j], gauss_kernel) * a_counts[j] + 1e-8
        # print(norm)
        e_a[i] = e_a_i / norm[i]

    for i in range(len(actions)):
        stats[i] = e_a[i] + C * np.sqrt(np.log(np.sum(norm)) / norm[i])

    return stats
