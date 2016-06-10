def updateWalkers(x, V, V1, w1, eps):
    # x contains a list of points of each walker
    # potential can tell you the potential evaluated at a point
    # eps is the time step we simulate for
    import numpy as np
    m = len(x)
    
    # first evolve operator e^{eps H} 
    # apply e^{-eps V / 2} weighting
    w = np.exp(-eps * V(x) / 2)
    w1 = w1 * np.exp(-eps * V1(x) / 2)
    # apply e^{ eps Delta} and re-approximate by 
    # deltas (just do random walk)
    x = x + np.sqrt(eps) * np.random.randn(m)
    # apply e^{-eps V / 2} weight again
    w = w * np.exp(-eps * V(x) / 2)
    w1 = w1 * np.exp(-eps * V1(x) / 2)

    val = np.mean(w)
    val1 = np.mean(w1)

    # normalize the weights
    w = w / np.sum(w)
    w1 = w1 / np.sum(w1)
    # now resample -- start with an empty list 
    # and then repopulate
    x2 = []
    w2 = []
    for i in range(m):
        p = np.random.rand()
        nj = int(m*w[i])

        if p < m*w[i] - np.floor(m*w[i]):
            nj = nj + 1

        for k in range(nj):
            x2.append(x[i])
            w2.append(w1[i] / w[i])

    return np.array(x2),np.array(w2),val,val1

if __name__ == "__main__":
    import numpy as np
    x = np.zeros(1000)
    w1 = np.ones(1000)
    eps = 0.01
    V = lambda x: x**2  / 2
    V1 = lambda x: x**2 * 1.05 / 2
    exact1 = np.sqrt(1.05)/2
    numiter = 10000
    values = np.zeros(numiter)
    values1 = np.zeros(numiter)
    ests = []
    ests1 = []
    for i in range(numiter):
        x,w1,val,val1 = updateWalkers(x, V, V1, w1, eps)
        values[i] = val
        values1[i] = val1
        halftime = np.ceil(i/2)
        if i % 200 == 50:
            est  = - np.log(np.mean(values[halftime:i])) / eps 
            est1 = - np.log(np.mean(values1[halftime:i])) / eps
            print '--'
            ests.append(est)
            ests1.append(est1)
            print est - 0.5
            print est1 - exact1
            print est-est1 - (0.5-exact1)
            - np.log(np.mean(values1[halftime:i])) / eps
            #print 1 / np.sqrt(len(x))
    import matplotlib.pyplot as plt
    ests = np.array(ests[10:])
    ests1 = np.array(ests1[10:])
    plt.plot(ests - 0.5)
    plt.plot(ests1-exact1, c='r')
    plt.plot(ests-ests1 - (0.5-exact1), c='g')
    plt.show()
