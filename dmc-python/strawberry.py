def updateWalkers(x, V, F, eps):
    # x contains a list of points of each walker
    # potential can tell you the potential evaluated at a point
    # eps is the time step we simulate for
    import numpy as np
    m = x.size

    # apply e^{-eps V / 2} weighting
    w = np.exp(-eps * V(x) / 2)
    # apply e^{ eps Delta} and re-approximate by 
    # deltas (just do random walk)
    x = x + np.sqrt(eps) * np.random.randn()
    # apply e^{-eps V / 2} weight again
    w = w * np.exp(-eps * V(x) / 2)

    val = np.mean(w)

    # normalize the weights
    w = w / np.sum(w)
    # now resample -- start with an empty list 
    # and then repopulate
    x2 = []
    for i in range(m):
        p = np.random.rand()
        nj = int(m*w[i])
        if p < m*w[i] - np.floor(m*w[i]):
            nj = nj + 1

        for k in range(nj):
            x2.append(x[i])

    return x2,val

if __name__ == "__main__":
    import numpy as np
    x = [0] * 500
    eps = 0.01
    V = lambda x: x**2 / 2
    numiter = 10000
    values = np.zeros(numiter)
    for i in range(numiter):
        x,val = updateWalkers(x, V, eps)
        values[i] = val
        halftime = np.ceil(i/2)
        if i % 100 == 50:
            print - np.log(np.mean(values[halftime:i])) / eps
            #print 1 / np.sqrt(len(x))
    import matplotlib.pyplot as plt
    n,bins,patches = plt.hist(x,20)
    plt.show()
