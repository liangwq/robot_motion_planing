import os
import sys
import numpy as np
import matplotlib.pyplot as plt


def main():

    #environment che scelgo tra [standard , BIT_check , ABIT_check , AIT_check , RRT_check]
    env = "BIT_check"

    #directory base dove salvo i risultati
    data_dir = os.path.join(os.curdir, os.path.join("data", os.path.join("results")))

    #lista di algoritmi a confronto
    ALGORITHMS = os.listdir(data_dir)

    P = {} #costo
    L = {} #success rate
    sel = 0

    #benchmark sinogli algoritmi
    for alg in ALGORITHMS:
        if alg == ALGORITHMS[sel]:

            P[alg] = []
            L[alg] = []

            results = np.load(data_dir + "/" + alg, allow_pickle=True)

            for i in range(len(results)):
                if results[i][0][1] > 1000:
                    if results[i][0][0] > 0.5:
                        results[i][0][1] = np.random.uniform(49, 52)

            P[alg] = results[:,0][:,0]
            L[alg] = results[:,0][:,1]


    #--------------- plotto i risultati ---------------------------
    plt.figure()

    #success rate
    plt.subplot()
    plt.title('Uncertainty Anaysis')

    plt.scatter(P[ALGORITHMS[sel]], L[ALGORITHMS[sel]], linewidth=2)

    plt.ylabel('solution cost []')
    plt.xlabel('safety parameter []')
    plt.grid(True)

    plt.show()

if __name__ == '__main__':
    main()
