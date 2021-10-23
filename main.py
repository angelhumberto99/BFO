import random as rand
import numpy as np
import matplotlib.pyplot as plt

def fitness(x):
    return np.sum(np.abs(np.array(x)))

def getCombAttract(h, Wr, X, d, Wa):
    res = 0
    for i in range(len(X)):
        res += (h * (np.exp(-Wr * (np.linalg.norm((X - X[i]), ord=2)**2)))
            - d * (np.exp(-Wa * (np.linalg.norm((X - X[i]), ord=2))**2)))
    return res

def plot(x, y, title):
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    plt1 = fig.add_subplot(1,1,1)
    plt1.plot(x, y)
    plt1.set_title(title)
    plt.show()

def main():
    c = 0.1 # step size
    N = 20 # population size
    Nc = 30 # chemotaxis steps
    Ns = 4 # cost reduction steps
    Nr = 4 # reproduction steps
    Ne = 2 # elimination-dispersal steps
    d = 1 # attraction force depth
    h = 1 # repulsion force depth
    Wa = 0.2 # attraction force width
    Wr = 10 # repulsion force width
    Pe = 0.25 # probability of elimination-dispersal
    best = float("inf")
    y = []
    x = []
    x_count = 0

    # population
    X = np.empty(N)
    for i in range(N):
        X[i] = rand.uniform(-10, 10)

    for l in range(Ne):
        for k in range(Nr):
            for j in range(Nc):
                for i in range(N):
                    c_attract_old = fitness(X[i]) + getCombAttract(h, Wr, X, d, Wa)
                    r = rand.random()
                    for m in range(Ns):
                        x_proof = X[i] + c * r
                        c_attract_new = fitness(x_proof) + getCombAttract(h, Wr, X, d, Wa)
                        if c_attract_new < c_attract_old:
                            X[i] = x_proof
                            if fitness(X[i]) < fitness(best):    
                                best = X[i]
                        else:
                            break
                    y.append(fitness(best))
                    x.append(x_count)
                    x_count += 1
            X = np.sort(X, axis=0)
            X[:int(np.floor(N/2))] = X[int(np.floor(N/2)):]
        for ind in range(N):
            r = rand.random()
            if r < Pe:
                X[i] = rand.uniform(-10, 10)
    print("Mejor solución: ", best)
    print("Mejor fitness: ", fitness(best))
    plot(x, y, "BFO")                    

if __name__ == "__main__":
    main()