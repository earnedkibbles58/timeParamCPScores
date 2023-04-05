

import numpy as np
# from gurobipy import *

import gurobipy as gp
from gurobipy import GRB

import time
import pickle
import random
import math
import matplotlib.pyplot as plt
from gurobipyTutorial import optimzeTimeAlphasKKT, optimzeTimeAlphasKKTNoMaxLowerBound


from runExpORCALSTMData import *



if __name__ == "__main__":

    save_path = "data/"

    fid = open(save_path + "d_value.pkl", 'rb')
    d_value = pickle.load(fid)
    fid.close()

    fid = open(save_path + "x_value.pkl", 'rb')
    x_value = pickle.load(fid)
    fid.close()

    fid = open(save_path + "y_value.pkl", 'rb')
    y_value = pickle.load(fid)
    fid.close()

    fid = open(save_path + "x_l_value.pkl", 'rb')
    x_l_value = pickle.load(fid)
    fid.close()

    fid = open(save_path + "y_l_value.pkl", 'rb')
    y_l_value = pickle.load(fid)
    fid.close()



    p_len = 20
    delta = 0.05

    random.seed(4568)

    dict_keys_x = list(x_value.keys())
    dict_key_x = dict_keys_x[0]
    print(dict_key_x)

    dict_keys_y = list(y_value.keys())
    dict_key_y = dict_keys_y[0]
    print(dict_key_y)

    dict_keys_x_l = list(x_l_value.keys())
    dict_key_x_l = dict_keys_x_l[0]
    print(dict_key_x_l)

    dict_keys_y_l = list(y_l_value.keys())
    dict_key_y_l = dict_keys_y_l[0]
    print(dict_key_y_l)


    p_len = 20
    delta = 0.05
    calib_percent = 0.5
    trace_for_alphas = 50
    makePlots = True


    dict_keys_x = list(x_value.keys())
    dict_key_x = dict_keys_x[0]
    print(dict_key_x)

    dict_keys_y = list(y_value.keys())
    dict_key_y = dict_keys_y[0]
    print(dict_key_y)

    dict_keys_x_l = list(x_l_value.keys())
    dict_key_x_l = dict_keys_x_l[0]
    print(dict_key_x_l)

    dict_keys_y_l = list(y_l_value.keys())
    dict_key_y_l = dict_keys_y_l[0]
    print(dict_key_y_l)


    keys_x = list(x_value[dict_key_x].keys())

    all_x = []
    all_y = []
    all_x_l = []
    all_y_l = []


    for k in keys_x:

        final_key = list(x_value[dict_key_x][k].keys())[0]

        x = x_value[dict_key_x][k][final_key]
        y = y_value[dict_key_y][k][final_key]
        x_l = x_l_value[dict_key_x_l][k][final_key]
        y_l = y_l_value[dict_key_y_l][k][final_key]

        all_x.append(x[p_len:2*p_len])
        all_y.append(y[p_len:2*p_len])

        all_x_l.append(x_l[0:p_len])
        all_y_l.append(y_l[0:p_len])


    num_trial = 100

    validation_coverages = []
    validation_coverages_union_bound = []
    validation_coverages_no_union_bound = []


    areas_over_time = [0]*p_len
    areas_over_time_union_bound = [0]*p_len
    areas_over_time_no_union_bound = [0]*p_len


    runtimes = []
    runtimes_milp = []
    obj_diffs = []
    alpha_soln_diffs = []

    for trial in range(num_trial):

        random.seed(trial**3)

        ## create train/test split
        temp = list(zip(all_x, all_y, all_x_l,all_y_l))
        random.shuffle(temp)

        res1,res2,res3,res4 = zip(*temp)
        all_x,all_y,all_x_l,all_y_l = list(res1),list(res2),list(res3),list(res4)

        calib_ind = round(len(all_x)*calib_percent)

        print(calib_ind)

        all_x_calib = all_x[0:calib_ind]
        all_y_calib = all_y[0:calib_ind]
        all_x_l_calib = all_x_l[0:calib_ind]
        all_y_l_calib = all_y_l[0:calib_ind]


        all_x_calib_alphas = all_x_calib[0:trace_for_alphas]
        all_y_calib_alphas = all_y_calib[0:trace_for_alphas]
        all_x_l_calib_alphas = all_x_l_calib[0:trace_for_alphas]
        all_y_l_calib_alphas = all_y_l_calib[0:trace_for_alphas]


        all_x_calib_CP = all_x_calib[trace_for_alphas:]
        all_y_calib_CP = all_y_calib[trace_for_alphas:]
        all_x_l_calib_CP = all_x_l_calib[trace_for_alphas:]
        all_y_l_calib_CP = all_y_l_calib[trace_for_alphas:]


        all_x_valid = all_x[calib_ind:]
        all_y_valid = all_y[calib_ind:]
        all_x_l_valid = all_x_l[calib_ind:]
        all_y_l_valid = all_y_l[calib_ind:]


        print("Num data calib: " + str(len(all_x_calib)))
        print("Num data valid: " + str(len(all_x_valid)))


        ## compute R values of data
        R_vals_calib_alpha = computeRValuesIndiv(all_x_calib_alphas,all_y_calib_alphas,all_x_l_calib_alphas,all_y_l_calib_alphas)

        ## run optimziation
        M = 100000 #big value for linearization of max constraint

        start_time = time.time()
        m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals_calib_alpha,delta,M)
        end_time = time.time()

        start_time_milp = time.time()
        m_milp = optimzeTimeAlphasKKT(R_vals_calib_alpha,delta,M)
        end_time_milp = time.time()


        print("Solve time: " + str(end_time-start_time))
        print("Solve time MILP: " + str(end_time_milp-start_time_milp))

        runtimes.append(end_time-start_time)
        runtimes_milp.append(end_time_milp-start_time_milp)

        alphas = []
        obj = 0
        for v in m.getVars():
            if "alphas" in v.varName:
                alphas.append(v.x)
            if "q" in v.varName:
                # print(v.x)
                obj = v.x
                print("obj: " + str(v.x))

        alphas_milp = []
        obj_milp = 0
        for v in m_milp.getVars():
            if "alphas" in v.varName:
                alphas_milp.append(v.x)
            if "q" in v.varName:
                # print(v.x)
                obj_milp = v.x
                print("obj MILP: " + str(v.x))

        running_diff = 0
        for alpha_ind in range(len(alphas)):
            running_diff += (alphas[alpha_ind]-alphas_milp[alpha_ind])**2
        running_diff = math.sqrt(running_diff)
        alpha_soln_diffs.append(running_diff)
        obj_diffs.append(abs(obj-obj_milp))

        ## run CP using alpha values on remaining calibration data
        D_cp = computeCPFixedAlphas(all_x_calib_CP,all_y_calib_CP,all_x_l_calib_CP,all_y_l_calib_CP,alphas,delta)
        validation_coverage = computeCoverageRAndAlphas(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,alphas,D_cp)


        ## run CP using circles and union bound
        delta_union = delta/p_len
        Ds_cp_union_bound = []
        Ds_cp_no_union_bound = []
        
        for t in range(p_len):
            x_vals = [all_x_calib[i][t] for i in range(len(all_x_calib))]
            y_vals = [all_y_calib[i][t] for i in range(len(all_x_calib))]
            x_hats = [all_x_l_calib[i][t] for i in range(len(all_x_calib))]
            y_hats = [all_y_l_calib[i][t] for i in range(len(all_x_calib))]

            D_val = computeCPCirlce(x_vals,y_vals,x_hats,y_hats,delta_union)
            Ds_cp_union_bound.append(D_val)

            D_val = computeCPCirlce(x_vals,y_vals,x_hats,y_hats,delta)
            Ds_cp_no_union_bound.append(D_val)

        ## compute trace wise coverage
        validation_coverage_union_bound = computeCoverageCircle(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,Ds_cp_union_bound)
        validation_coverage_no_union_bound = computeCoverageCircle(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,Ds_cp_no_union_bound)


        validation_coverages.append(validation_coverage)
        validation_coverages_union_bound.append(validation_coverage_union_bound)
        validation_coverages_no_union_bound.append(validation_coverage_no_union_bound)


        for t in range(p_len):

            ## compute area of each conformal region

            area = math.pi*(D_cp/alphas[t])**2
            area_union_bound = math.pi*Ds_cp_union_bound[t]**2
            area_no_union_bound = math.pi*Ds_cp_no_union_bound[t]**2

            areas_over_time[t] += area/num_trial
            areas_over_time_union_bound[t] += area_union_bound/num_trial
            areas_over_time_no_union_bound[t] += area_no_union_bound/num_trial
        

    print("Validation coverage: " + str(np.mean(validation_coverages)))
    print("Validation coverage union: " + str(np.mean(validation_coverages_union_bound)))
    print("Validation coverage no union: " + str(np.mean(validation_coverages_no_union_bound)))



    image_save_dir = "images/forPaper/"
    ## plot validation errs and prediction regions
    for i in range(len(all_x_valid[0])):

        # individual histograms
        plt.clf()
        plt.hist(validation_coverages,bins=25)
        plt.xlabel("Validation Coverage",fontsize="16")
        plt.ylabel("Counts",fontsize="16")
        plt.savefig(image_save_dir + "coverageHistAlphas.png")

        plt.clf()
        plt.hist(validation_coverages_union_bound,bins=25)
        plt.xlabel("Validation Coverage",fontsize="16")
        plt.ylabel("Counts",fontsize="16")
        plt.savefig(image_save_dir + "coverageHistUnionBound.png")

        plt.clf()
        plt.hist(validation_coverages_no_union_bound,bins=25)
        plt.xlabel("Validation Coverage",fontsize="16")
        plt.ylabel("Counts",fontsize="16")
        plt.savefig(image_save_dir + "coverageHistNoUnionBoud.png")

        plt.clf()


        plt.plot(areas_over_time,'g*',label="Our approach")
        plt.plot(areas_over_time_union_bound,'r*',label="Union bound")
        # plt.plot(areas_over_time_no_union_bound,'k*',label="No union bound")
        plt.legend(fontsize="16")


        plt.ylabel("Average CP Region Area",fontsize="16")
        plt.xlabel("Prediction Lookahead Time",fontsize="16")

        ax=plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig(image_save_dir + "areasOfRegions.png")

    

        # all histograms at once
        plt.clf()

        plt.hist(validation_coverages,bins=25,label="Our approach",alpha=0.5)
        plt.hist(validation_coverages_union_bound,bins=25,label="Union bound",alpha=0.5)
        plt.xlabel("Validation Coverage",fontsize="16")
        plt.ylabel("Counts",fontsize="16")
        plt.legend(loc="upper left",fontsize="16")
        ax=plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig(image_save_dir + "allHistsInOne.png")



    print("Average runtime: " + str(np.average(runtimes)))
    print("Average runtime MILP: " + str(np.average(runtimes_milp)))

    print("Average aplha L2 diff: " + str(np.average(alpha_soln_diffs)))
    print("Max alpha L2 diff: " + str(max(alpha_soln_diffs)))


    print("Average obj diff:" + str(np.average(obj_diffs)))
    print("Max obj diff: " + str(np.average(obj_diffs)))