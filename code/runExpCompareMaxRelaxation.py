import numpy as np
# from gurobipy import *

import gurobipy as gp
from gurobipy import GRB

import time

import matplotlib.pyplot as plt

from gurobipyTutorial import optimzeTimeAlphasKKT, optimzeTimeAlphasKKTNoMaxLowerBound


NUM_TRIAL = 10





if __name__ == "__main__":
    
    n = 100 # number of datapoints
    T = 10 # length of time series

    M = 100000 #big value for linearization of max constraint

    delta = 0.05

    solve_times_full_max = []
    solve_times_loose_max = []

    q_vals_full_max = []
    q_vals_loose_max = []

    alpha_vals_diff = []

    for i in range(NUM_TRIAL):

        np.random.seed(i)
        fake_R_vals = np.random.rand(n,T)


        start_time = time.time()
        m_full = optimzeTimeAlphasKKT(fake_R_vals,delta,M)
        end_time = time.time()
        solve_times_full_max.append(end_time-start_time)


        start_time = time.time()
        m_loose = optimzeTimeAlphasKKTNoMaxLowerBound(fake_R_vals,delta,M)
        end_time = time.time()
        solve_times_loose_max.append(end_time-start_time)

        temp_alphas_full = []
        temp_alphas_loose = []

        for v in m_full.getVars():
            if v.varName == "q":
                q_vals_full_max.append(v.x)

            if "alphas" in v.varName:
                temp_alphas_full.append(v.x)

        for v in m_loose.getVars():
            if v.varName == "q":
                q_vals_loose_max.append(v.x)

            if "alphas" in v.varName:
                temp_alphas_loose.append(v.x)
                
        temp_abs_alpha_diff = 0
        for j in range(len(temp_alphas_full)):
            temp_abs_alpha_diff += abs(temp_alphas_full[j]-temp_alphas_loose[j])
        
        alpha_vals_diff.append(temp_abs_alpha_diff)


    plt.plot(solve_times_full_max,'r*',label="Full max")
    plt.plot(solve_times_loose_max,'b*', label="Loose max")
    plt.legend()
    plt.ylabel("Runtime (s)")
    plt.xlabel("Trial")
    plt.savefig("maxComparisonRunTimes.png")
    plt.clf()

    plt.plot(q_vals_full_max,'r*',label="Full max")
    plt.plot(q_vals_loose_max,'b*', label="Loose max")
    plt.legend()
    plt.ylabel("Quantile value")
    plt.xlabel("Trial")
    plt.savefig("maxComparisonQValues.png")
    plt.clf()

    plt.plot(alpha_vals_diff,'b*')
    plt.ylabel("Sum of absolute differences of alpha values (L1 norm)")
    plt.xlabel("Trial")
    plt.savefig("maxComparisonAlphaDiffs.png")
    plt.clf()

