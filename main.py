import numpy as np
import galois
import matplotlib.pyplot as plt
from landslide import *
import copy
import math

# check if two binary vectors are equal
def check_equal(c, y):
    mrg = c + y
    mrg = np.array(mrg)
    if(np.sum(mrg) > 0): # if the sum evaluates to <= 1
        return False
    else: # if the sum evaluates to 0
        return True

# demodulate the input signal R
def demodulate(R):
    y = [int(x < 0) for x in R]
    return y

# Compute the LLR of a rec'd bit
def bitLLR(ri, sigma):
    return (2 * np.abs(ri)) / (sigma**2)

# Return the probability that a rec'd bit was flipped by noise
def getPFlip(ri, sigma):
    num = np.exp(-bitLLR(ri, sigma))
    den = 1 + np.exp(-bitLLR(ri, sigma))
    return num / den

# Return the probability that each bit in the rec'd signal was flipped
def getPFlipString(signal, sigma):
    l = [getPFlip(signal[i], sigma) for i in range(len(signal))]
    return l

# Return the probability of a computed noise effect sequence
def getProbNoiseEffect(pFlipString, N):
    p = 1
    for i in range(len(pFlipString)):
        if(not (i in N)):
            p = p * (1 - pFlipString[i])
        else:
            p = p * pFlipString[i]
    return p

# Estimate the probability that this decoding is correct
def estimateProbCorrect(prob, aggProb, k, n):
    denominator = (1 - aggProb)*((2**k - 1)/(2**n - 1)) + prob
    res = prob / denominator
    if(res < 0):
        print("prob:", prob, "aggprob:", aggProb, "phi:", res)
    return res


# use ORBGRAND principles and the landslide algorithm
# to estimate the decoding of the input signal
def orbgrand_decode(y, I, H, signal, sigma):
    noise = []
    n = len(y)

    nQueries = 0
    aggProb = 0
    probFlipString = getPFlipString(signal, sigma)

    for W in range(0, 1 + n*(n+1)//2):
        w_lb = np.ceil((1+2*n-math.sqrt((1+2*n)**2 - 8*W))/2)
        w_ub = np.floor((math.sqrt(1 + 8*W) - 1)/2)
        for w in range(int(w_lb), int(w_ub)+1):
            for effect in landslide(W, w, n):
                c = copy.deepcopy(y)
                noise = I[effect]

                noiseProb = getProbNoiseEffect(probFlipString, noise)
                aggProb = aggProb + noiseProb
                nQueries = nQueries + 1

                for idx in noise:
                    c[idx] = c[idx] ^ 1
                    codebook_check = H.T @ c
                    if(np.sum(codebook_check.view(np.ndarray)) == 0):
                        return c, noise, {'queries': nQueries, 'prob': noiseProb, 'aggProb': aggProb}

if __name__ == '__main__':
    n = 128
    k = 120

    GF = galois.GF(2)

    I = np.identity(k, dtype=int)
    P = np.random.randint(2, size=(n-k, k), dtype=int)

    # Generator Matrix
    G = GF(np.concatenate((I, P), axis=0))
    # Parity Check Matrix
    H = GF(np.concatenate((P.T, np.identity(n-k, dtype=int))))
    assert(G.shape == (n, k))

    # codeword
    u = GF(np.random.randint(2, size=(k)))

    # Number of transmissions to simulate
    N_TX = 10**4

    sigma = 0.486914439648981

    a = []
    b = []

    it = 0
    for i in range(N_TX):
        it = it + 1

        # Random input
        u = GF(np.random.randint(2, size=(k)))

        # Encoded input
        x = u@G.T
        x = x.view(np.ndarray)

        # Noise effect
        N = np.random.normal(0, sigma, n)

        # Output of channel
        R = []
        for i in range(len(x)):
            R.append(1 - 2*x[i])
            R[i] = R[i] + N[i]

        # Output of channel sorted by reliability
        I = np.argsort(np.abs(R))

        # Hard demodulated signal
        y = demodulate(R)

        # Decoded code word, associated noise effect, soft output information
        c_hat, noise, soft_output = orbgrand_decode(GF(y), I, H, R, sigma)

        # Estimated probability that this decoding is correct
        phi = estimateProbCorrect(soft_output["prob"], soft_output["aggProb"], k, n)


        a.append(phi)
        if(not check_equal(x.view(GF), c_hat)):
            b.append(0)
        else:
            b.append(1)
        if(it % 1000 == 0):
            # Sanity check
            print("Iterations:", it)

    # Information for verifying effectiveness of this implementation
    j = range(1, 21)
    Ija = dict.fromkeys(list(j), 0)
    Ijb = dict.fromkeys(list(j), 0)
    IjCount = dict.fromkeys(list(j), 0)

    for i in range(len(a)):
        ai = a[i]
        bi = b[i]
        idx = np.ceil(ai*20)
        if(idx == 0):
            idx = 1
        Ija[idx] = Ija[idx] + a[i]
        Ijb[idx] = Ijb[idx] + b[i]
        IjCount[idx] = IjCount[idx] + 1

    x = [Ija[i]/(IjCount[i]) for i in Ija.keys()]
    y = [Ijb[i]/(IjCount[i]) for i in Ijb.keys()]

    # Plotting Stuff
    plt.plot(x, y, '-x', label="ORBGRAND RLC[128, 120]")
    plt.title("Predicted probability a decoding is correct versus empirical probability that it is correct.")
    plt.xlabel("Predicted Probability Correct")
    plt.ylabel("Estimated conditional probability correct")
    plt.grid(True, which="both")
    plt.plot([0, 1], [0, 1], '--', label="x=y")
    plt.legend()

    plt.savefig("pcorrectvsestimated.png")

    plt.show()
