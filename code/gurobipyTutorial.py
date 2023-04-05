

import numpy as np
import math
# from gurobipy import *

import gurobipy as gp
from gurobipy import GRB

import time

def firstEx():
    try:

        # Create a new model
        m = gp.Model("mip1")
        # Create variables
        x = m.addVar(vtype=GRB.BINARY, name="x")
        y = m.addVar(vtype=GRB.BINARY, name="y")
        z = m.addVar(vtype=GRB.BINARY, name="z")
        # Set objective
        m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
        # Add constraint: x + 2 y + 3 z <= 4
        m.addConstr(x + 2 * y + 3 * z <= 4, "c0")
        # Add constraint: x + y >= 1
        m.addConstr(x + y >= 1, "c1")
        m.optimize()
        for v in m.getVars():
            print(v.varName, v.x)
        
        print('Obj: ' + str(m.objVal))





    except gp.GurobiError as e:
        print('Error from gurobi: ' + str(e))
    





def optimzeTimeAlphasMinArgminHeuristic(R_vals,delta,M):

    n = len(R_vals)
    T = len(R_vals[0])

    ## R_vals: n x T dim array. n data points, each with T non-conformity scores



    try:
        m = gp.Model("mip1")



        ## declare variables we have
        # q, continuous, positive
        # alpha_t, continuous, positive, one for each of T time steps in time series
        # e^+_i, e^-_i, continuous, positive, one for each of n time series
        # b_t^i, binary, one for each of T time steps for each of n time series (T x n total)
        # R^i, continuous, positive, one for each of n time series

        es_plus = m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="es_plus")
        es_minus = m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="es_minus")
        b = m.addVars(n,T,vtype=GRB.BINARY, name="b")
        Rs = m.addVars(n,vtype=GRB.CONTINUOUS, name="Rs")

        q = m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="q")
        alphas = m.addVars(T,lb=0,vtype=GRB.CONTINUOUS, name="alphas")

        ## create objective
        obj = gp.LinExpr()

        for i in range(n):
            obj += (1-delta)*es_plus[i]
            obj += delta*es_minus[i]
        m.setObjective(obj,GRB.MINIMIZE)

        ## declare constraints we have
        for i in range(n):
            m.addConstr(es_plus[i]-es_minus[i] == Rs[i]-q)
            m.addConstr(es_plus[i] >= 0)
            m.addConstr(es_minus[i] >= 0)
        
        for i in range(n):
            for t in range(T):
                m.addConstr(Rs[i] >= alphas[t]*R_vals[i][t])

                m.addConstr(Rs[i] <= alphas[t]*R_vals[i][t] + (1-b[(i,t)])*M)
        

        
        for i in range(n):
            b_constraint = gp.LinExpr()
            for t in range(T):
                b_constraint += b[(i,t)]
            m.addConstr( b_constraint == 1 )
        
        m_constraint = gp.LinExpr()
        for t in range(T):
            m_constraint += alphas[t]
            m.addConstr(alphas[t] >= 0)
        m.addConstr(m_constraint == 1)
        







        m.optimize()
        for v in m.getVars():
            if v.varName == "q" or "alphas" in v.varName:
                print(v.varName, v.x)
        
        print('Obj: ' + str(m.objVal))


    except gp.GurobiError as e:
        print('Error from gurobi: ' + str(e))







def optimzeTimeAlphasKKT(R_vals,delta,M):

    n = len(R_vals)
    T = len(R_vals[0])

    ## R_vals: n x T dim array. n data points, each with T non-conformity scores



    try:
        m = gp.Model("mip1")



        ## declare variables we have
        # q, continuous, positive
        # alpha_t, continuous, positive, one for each of T time steps in time series
        # e^+_i, e^-_i, continuous, positive, one for each of n time series
        # b_t^i, binary, one for each of T time steps for each of n time series (T x n total)
        # R^i, continuous, positive, one for each of n time series

        es_plus = m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="es_plus")
        es_minus = m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="es_minus")
        b = m.addVars(n,T,vtype=GRB.BINARY, name="b")
        Rs = m.addVars(n,vtype=GRB.CONTINUOUS, name="Rs")

        us_plus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="us_plus")
        us_minus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="us_minus")
        v = m.addVars(n, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="v")


        q = m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="q")
        alphas = m.addVars(T,lb=0,vtype=GRB.CONTINUOUS, name="alphas")


        ## create objective
        obj = gp.LinExpr(q)
        m.setObjective(obj,GRB.MINIMIZE)

        #### KKT constraints

        ## gradients are 0 (stationary condition)
        q_gradient_constraint = gp.LinExpr()
        for i in range(n):
            m.addConstr((1-delta) - us_plus[i] + v[i] == 0)
            m.addConstr(delta - us_minus[i] - v[i] == 0)
            q_gradient_constraint += (v[i])
        m.addConstr(q_gradient_constraint == 0)

        ## complementary slackness
        for i in range(n):
            m.addConstr(us_plus[i]*es_plus[i] == 0)
            m.addConstr(us_minus[i]*es_minus[i] == 0)
        
        # ## primal feasibility
        for i in range(n):
            m.addConstr(es_plus[i] + q - es_minus[i] - Rs[i] == 0)
            m.addConstr(es_plus[i] >= 0)
            m.addConstr(es_minus[i] >= 0)
        
        # ## dual feasibility
        for i in range(n):
            m.addConstr(us_plus[i] >= 0)
            m.addConstr(us_minus[i] >= 0)
        


        # ## declare constraints we have
        # for i in range(n):
        #     m.addConstr(es_plus[i]-es_minus[i] == Rs[i]-q)
        #     m.addConstr(es_plus[i] >= 0)
        #     m.addConstr(es_minus[i] >= 0)
        
        for i in range(n):
            for t in range(T):
                m.addConstr(Rs[i] >= alphas[t]*R_vals[i][t])
                m.addConstr(Rs[i] <= alphas[t]*R_vals[i][t] + (1-b[(i,t)])*M)
        

        
        for i in range(n):
            b_constraint = gp.LinExpr()
            for t in range(T):
                b_constraint += b[(i,t)]
            m.addConstr( b_constraint == 1 )
        
        m_constraint = gp.LinExpr()
        for t in range(T):
            m_constraint += alphas[t]
            m.addConstr(alphas[t] >= 0)
        m.addConstr(m_constraint == 1)
        







        m.optimize()
        for v in m.getVars():
            if v.varName == "q" or "alphas" in v.varName:
                print(v.varName, v.x)
        
        print('Obj: ' + str(m.objVal))


    except gp.GurobiError as e:
        print('Error from gurobi: ' + str(e))

    return m





def optimzeTimeAlphasKKTNoMaxLowerBound(R_vals,delta,M):

    n = len(R_vals)
    T = len(R_vals[0])

    ## R_vals: n x T dim array. n data points, each with T non-conformity scores



    try:
        m = gp.Model("mip1")



        ## declare variables we have
        # q, continuous, positive
        # alpha_t, continuous, positive, one for each of T time steps in time series
        # e^+_i, e^-_i, continuous, positive, one for each of n time series
        # b_t^i, binary, one for each of T time steps for each of n time series (T x n total)
        # R^i, continuous, positive, one for each of n time series

        es_plus = m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="es_plus")
        es_minus = m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="es_minus")
        Rs = m.addVars(n,vtype=GRB.CONTINUOUS, name="Rs")

        us_plus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="us_plus")
        us_minus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="us_minus")
        v = m.addVars(n, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="v")


        q = m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="q")
        alphas = m.addVars(T,lb=0,vtype=GRB.CONTINUOUS, name="alphas")


        m.params.NonConvex = 2

        ## create objective
        obj = gp.LinExpr(q)
        m.setObjective(obj,GRB.MINIMIZE)

        #### KKT constraints

        ## gradients are 0 (stationary condition)
        q_gradient_constraint = gp.LinExpr()
        for i in range(n):
            m.addConstr((1-delta) - us_plus[i] + v[i] == 0)
            m.addConstr(delta - us_minus[i] - v[i] == 0)
            q_gradient_constraint += (v[i])
        m.addConstr(q_gradient_constraint == 0)

        ## complementary slackness
        for i in range(n):
            m.addConstr(us_plus[i]*es_plus[i] == 0)
            m.addConstr(us_minus[i]*es_minus[i] == 0)
        
        # ## primal feasibility
        for i in range(n):
            m.addConstr(es_plus[i] + q - es_minus[i] - Rs[i] == 0)
            m.addConstr(es_plus[i] >= 0)
            m.addConstr(es_minus[i] >= 0)
        
        # ## dual feasibility
        for i in range(n):
            m.addConstr(us_plus[i] >= 0)
            m.addConstr(us_minus[i] >= 0)
        

        
        for i in range(n):
            for t in range(T):
                m.addConstr(Rs[i] >= alphas[t]*R_vals[i][t])
        
        
        m_constraint = gp.LinExpr()
        for t in range(T):
            m_constraint += alphas[t]
            m.addConstr(alphas[t] >= 0)
        m.addConstr(m_constraint == 1)
        


        m.optimize()
        # for v in m.getVars():
            # if v.varName == "q" or "alphas" in v.varName:
            #     print(v.varName, v.x)
        
        # print('Obj: ' + str(m.objVal))


    except gp.GurobiError as e:
        print('Error from gurobi: ' + str(e))

    return m



def optimzeTimeAlphasKKTNoMaxLowerBoundMinArea(R_vals,delta,M):

    n = len(R_vals)
    T = len(R_vals[0])

    ## R_vals: n x T dim array. n data points, each with T non-conformity scores



    try:
        m = gp.Model("mip1")



        ## declare variables we have
        # q, continuous, positive
        # alpha_t, continuous, positive, one for each of T time steps in time series
        # e^+_i, e^-_i, continuous, positive, one for each of n time series
        # b_t^i, binary, one for each of T time steps for each of n time series (T x n total)
        # R^i, continuous, positive, one for each of n time series

        es_plus = m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="es_plus")
        es_minus = m.addVars(n,lb=0,vtype=GRB.CONTINUOUS, name="es_minus")
        Rs = m.addVars(n,vtype=GRB.CONTINUOUS, name="Rs")

        us_plus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="us_plus")
        us_minus = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name="us_minus")
        v = m.addVars(n, lb=-float('inf'), vtype=GRB.CONTINUOUS, name="v")


        q = m.addVar(lb=0,vtype=GRB.CONTINUOUS, name="q")
        alphas = m.addVars(T,lb=0,vtype=GRB.CONTINUOUS, name="alphas")


        m.params.NonConvex = 2

        ## create objective
        obj = gp.LinExpr()
        for t in range(T):
            obj += math.pi * (q/alphas[t])**2
        m.setObjective(obj,GRB.MINIMIZE)

        #### KKT constraints

        ## gradients are 0 (stationary condition)
        q_gradient_constraint = gp.LinExpr()
        for i in range(n):
            m.addConstr((1-delta) - us_plus[i] + v[i] == 0)
            m.addConstr(delta - us_minus[i] - v[i] == 0)
            q_gradient_constraint += (v[i])
        m.addConstr(q_gradient_constraint == 0)

        ## complementary slackness
        for i in range(n):
            m.addConstr(us_plus[i]*es_plus[i] == 0)
            m.addConstr(us_minus[i]*es_minus[i] == 0)
        
        # ## primal feasibility
        for i in range(n):
            m.addConstr(es_plus[i] + q - es_minus[i] - Rs[i] == 0)
            m.addConstr(es_plus[i] >= 0)
            m.addConstr(es_minus[i] >= 0)
        
        # ## dual feasibility
        for i in range(n):
            m.addConstr(us_plus[i] >= 0)
            m.addConstr(us_minus[i] >= 0)
        

        
        for i in range(n):
            for t in range(T):
                m.addConstr(Rs[i] >= alphas[t]*R_vals[i][t])
        
        
        m_constraint = gp.LinExpr()
        for t in range(T):
            m_constraint += alphas[t]
            m.addConstr(alphas[t] >= 0)
        m.addConstr(m_constraint == 1)
        


        m.optimize()
        # for v in m.getVars():
            # if v.varName == "q" or "alphas" in v.varName:
            #     print(v.varName, v.x)
        
        # print('Obj: ' + str(m.objVal))


    except gp.GurobiError as e:
        print('Error from gurobi: ' + str(e))

    return m



if __name__ == "__main__":
    # firstEx()

    np.random.seed(1)

    n = 100 # number of datapoints
    T = 10 # length of time series

    M = 100000 #big value for linearization of max constraint

    fake_R_vals = np.random.rand(n,T)
    delta = 0.05

    start_time = time.time()
    # optimzeTimeAlphasMinArgminHeuristic(fake_R_vals,delta,M)
    # optimzeTimeAlphasKKT(fake_R_vals,delta,M)
    optimzeTimeAlphasKKTNoMaxLowerBound(fake_R_vals,delta,M)
    end_time = time.time()

    print("Solve time: " + str(end_time-start_time))
    



    # for i in range(n):
    #     for t in range(T):
    #         fake_R_vals[i][t] = (t+1)*fake_R_vals[i][t]
    

    # optimzeTimeAlphasMinArgminHeuristic(fake_R_vals,delta,M)

    # optimzeTimeAlphasKKT(fake_R_vals,delta,M)
