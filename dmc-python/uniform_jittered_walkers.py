def updateWalkers(x, V, eps):
    # x contains a list of points of each walker
    # potential can tell you the potential evaluated at a point
    # eps is the time step we simulate for
    import numpy as np
    m = len(x)
    
    # first evolve operator e^{eps H} 
    # apply e^{-eps V / 2} weighting
    w = np.exp(-eps * V(x) / 2)
    # apply e^{ eps Delta} and re-approximate by 
    # deltas (just do random walk)
    x = x + np.sqrt(eps) * np.random.randn(m)
    # apply e^{-eps V / 2} weight again
    w = w * np.exp(-eps * V(x) / 2)

    val = np.mean(w)

    # normalize the weights
    w = w / np.sum(w)
    # now resample -- start with an empty list 
    # and then repopulate
    markers = np.cumsum(w) 
    lattice = np.linspace(0,1,m+1)[:m] + np.random.rand(m)*1.0/m
    x2 = []
    # index of walker interval
    Widx = 0
    for i in xrange(m):
        # i is index of lattice point
        while lattice[i] > markers[Widx]:
            Widx = Widx + 1
        x2.append(x[Widx])
    return np.array(x2),val


if __name__ == "__main__":
    import numpy as np
    x = np.zeros(1600)
    eps = 0.01
    V = lambda x: x**2 * 1.0 / 2
    numiter = 50000
    values = np.zeros(numiter)
    ests = []
    for i in range(numiter):
        x,val = updateWalkers(x, V,eps)
        values[i] = val
        halftime = np.ceil(i/2)
        est  = - np.log(np.mean(values[halftime:i])) / eps 
        ests.append(est)
        if i % 200 == 10:
            print ests[-1] - 0.5
    import matplotlib.pyplot as plt
    ests = np.array(ests[1000:])
    plt.plot(ests - 0.5)
    plt.show()
