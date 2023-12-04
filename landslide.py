import numpy as np

'''
ChatGPT is a co-author of this translation from landslide.m
'''

def landslide(W, w, n):
    if W == 0:
        return []
    elif W == n * (n + 1) / 2:
        return [list(range(0, n))]

    W1 = W - w * (w + 1) / 2
    n1 = n - w

    # Create the first integer partition
    jj = 1
    u = np.zeros(w, dtype=int)
    k = 1
    u = mountain_build(u, k, w, W1, n1)
    z = [u.copy()]

    # Evaluate drops
    d = np.roll(u, -1) - u
    d[-1] = 0

    # Evaluate accumulated drops
    D = np.cumsum(d[::-1])[::-1]

    # Each loop generates a new integer partition
    while D[0] >= 2:
        # Find the last index with an accumulated drop >=2
        k = np.where(D >= 2)[0][-1]

        # Increase its index by one.
        u[k] = u[k] + 1
        u = mountain_build(u, k, w, W1, n1)

        # Record the partition
        jj += 1
        z.append(u.copy())

        # Evaluate drops
        d = np.roll(u, -1) - u
        d[-1] = 0

        # Evaluate accumulated drops
        D = np.cumsum(d[::-1])[::-1]

    z = np.array(z) + np.tile(np.arange(0, w), (len(z), 1))
    return z.tolist()


def mountain_build(u, k, w, W1, n1):
    if w == 1:
        # Special case when w is 1
        u[0] = int(W1)
    else:
        u[k:w] = u[k] * np.ones(w - k, dtype=int)
        W2 = W1 - np.sum(u)
        q = int(W2 // (n1 - u[k]))
        r = int(W2 % (n1 - u[k]))

        if q != 0:
            u[w - q:w] = n1 * np.ones(q, dtype=int)

        if w - q > 0:
            u[w - q - 1] = u[w - q - 1] + r

    return u