import numpy as np
import string
from matplotlib import pyplot as plt
from numba import jit


@jit(nopython=True)
def viterbi(o, a, b, pi):
    """Computes the most likely sequence of hidden states S given
    data o and parameters a, b, pi.  S = S* in the lectures."""
    T = len(o)
    n = len(a)

    ell = np.zeros(shape=(n, T))
    for i in range(0, n):
        ell[i, 0] = np.log(pi[i] * b[i, o[0]])

    phi = np.zeros(shape=(T, n), dtype=np.int32)
    for t in range(0, T - 1):
        for j in range(0, n):
            to_be_maxed = np.zeros(n)
            for i in range(0, n):
                to_be_maxed[i] = ell[i, t] + np.log(a[i, j])
            phi[t + 1, j] = np.argmax(to_be_maxed)
            ell[j, t + 1] = to_be_maxed[phi[t + 1, j]]\
                + np.log(b[j, o[t + 1]])

    S = np.zeros(T, dtype=np.int32)
    S[-1] = np.argmax(ell[:, -1])
    for t in range(2, T + 1):
        S[-t] = phi[-t + 1, S[-t + 1]]

    return S


if __name__ == "__main__":
    o = np.loadtxt('observations.txt', dtype=np.int32)
    a = np.loadtxt('transitionMatrix.txt', dtype=np.float64)
    b = np.loadtxt('emissionMatrix.txt', dtype=np.float64)
    pi = np.loadtxt('initialStateDistribution.txt', dtype=np.float64)

    # Viterbi path
    S = viterbi(o, a, b, pi)

    # Checking answer
    alphabet = string.ascii_uppercase + ' '
    message = ''
    for t in range(0, len(S)):
        if t == 0:
            message += alphabet[S[t]]
        elif S[t] != S[t - 1]:
            message += alphabet[S[t]]

    print(message)

    # Plotting S versus t
    t = np.array(range(0, len(S)))
    plt.plot(t, S)
    plt.xlabel('$t$')
    plt.ylabel('$S_t^*$')
    plt.title('Most likely sequence $S^*$ of hidden states versus time')
    plt.show()
