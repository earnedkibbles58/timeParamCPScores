

import numpy as np
# from gurobipy import *

import gurobipy as gp
from gurobipy import GRB

import time
import pickle
import random
import math
import matplotlib.pyplot as plt
from gurobipyTutorial import optimzeTimeAlphasKKT, optimzeTimeAlphasKKTNoMaxLowerBound, optimzeTimeAlphasKKTNoMaxLowerBoundMinArea


PLOT_VALIDATION_TRACES = True
NUM_VALID_TO_PLOT = 100

def computeRValuesIndiv(x_vals,y_vals,x_hats,y_hats):

    R_vals = [ [math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))]  for i in range(len(x_vals))  ]

    return R_vals



def computeCPCirlce(x_vals,y_vals,x_hats,y_hats,delta):

    R_vals = [ math.sqrt((x_vals[i]-x_hats[i])**2 + (y_vals[i]-y_hats[i])**2) for i in range(len(x_vals))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return(R_vals[ind_to_ret])



def computeCPFixedAlphas(x_vals,y_vals,x_hats,y_hats,alphas,delta):

    R_vals = [max([alphas[j]*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i])) ]) for i in range(len(x_vals))]
    
    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return R_vals[ind_to_ret]



def computeCoverageRAndAlphas(x_vals,y_vals,x_hats,y_hats,alphas,D_cp):

    R_vals = [max([alphas[j]*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i])) ]) for i in range(len(x_vals))]

    num_points_within = sum(r <= D_cp for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct


def computeCoverageCircle(x_vals,y_vals,x_hats,y_hats,Ds_cp):
    coverage_count = 0
    for i in range(len(x_vals)):

        temp = sum([1 if math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) > Ds_cp[j] else 0 for j in range(len(x_vals[i]))])
        if temp == 0:
            coverage_count += 1
        

    coverage_pct = float(coverage_count)/len(x_vals)
    return coverage_pct


def plot_circle(x, y, size, color="-b", label=None):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]

    if label is None:
        plt.plot(xl, yl, color)
    else:
        plt.plot(xl, yl, color,label=label)



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

    random.seed(34956)

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

    all_x_err = []
    all_y_err = []


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


    ## create train/test split
    temp = list(zip(all_x, all_y, all_x_l,all_y_l))
    random.shuffle(temp)

    res1,res2,res3,res4 = zip(*temp)
    all_x,all_y,all_x_l,all_y_l = list(res1),list(res2),list(res3),list(res4)

    calib_ind = round(len(all_x)*calib_percent)

    print("Num calibration data: " + str(calib_ind))
    print("Num validation data: " + str(len(all_x)-calib_ind))

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

    
    # optimzeTimeAlphasMinArgminHeuristic(fake_R_vals,delta,M)
    

    start_time = time.time()
    m = optimzeTimeAlphasKKTNoMaxLowerBound(R_vals_calib_alpha,delta,M)
    # m_min_area = optimzeTimeAlphasKKTNoMaxLowerBoundMinArea(R_vals_calib_alpha,delta,M)
    end_time = time.time()


    start_time_milp = time.time()
    m_milp = optimzeTimeAlphasKKT(R_vals_calib_alpha,delta,M)
    end_time_milp = time.time()

    # start_time_min_area = time.time()
    # m_min_area = optimzeTimeAlphasKKTNoMaxLowerBoundMinArea(R_vals_calib_alpha,delta,M)
    # end_time_min_area = time.time()


    print("Solve time: " + str(end_time-start_time))
    print("Solve time MILP: " + str(end_time_milp-start_time_milp))
    # print("Solve time min area: " + str(end_time_min_area-start_time_min_area))

    alphas = []    
    for v in m.getVars():
        if "alphas" in v.varName:
            alphas.append(v.x)
        if "q" in v.varName:
            # print(v.x)
            print("obj: " + str(v.x))

    alphas_milp = []    
    for v in m_milp.getVars():
        if "alphas" in v.varName:
            alphas_milp.append(v.x)
        if "q" in v.varName:
            # print(v.x)
            print("obj MILP: " + str(v.x))


    print("alphas: " + str(alphas))
    print("alphas MILP: " + str(alphas_milp))

    plt.plot(alphas,'k*')
    plt.xlabel("Prediction Horizon",fontsize="16")
    plt.ylabel("Alpha Value",fontsize="16")

    plt.savefig('images/forPaper/alphas.png')



    # alphas_min_area = []    
    # for v in m_min_area.getVars():
    #     if "alphas" in v.varName:
    #         alphas_min_area.append(v.x)

    # print(alphas_min_area)

    # plt.plot(alphas_min_area,'k*')
    # plt.xlabel("Prediction Horizon")
    # plt.ylabel("Alpha Value")

    # plt.savefig('images/exp1/alphas_min_area.png')




    ## run CP using alpha values on remaining calibration data
    D_cp = computeCPFixedAlphas(all_x_calib_CP,all_y_calib_CP,all_x_l_calib_CP,all_y_l_calib_CP,alphas,delta)
    validation_coverage = computeCoverageRAndAlphas(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,alphas,D_cp)

    # D_cp_min_area = computeCPFixedAlphas(all_x_calib_CP,all_y_calib_CP,all_x_l_calib_CP,all_y_l_calib_CP,alphas_min_area,delta)
    # validation_coverage_min_area = computeCoverageRAndAlphas(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,alphas_min_area,D_cp_min_area)

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


    print("Validation coverage: " + str(validation_coverage))
    # print("Validation coverage min area: " + str(validation_coverage_min_area))
    print("Validation coverage union: " + str(validation_coverage_union_bound))
    print("Validation coverage no union: " + str(validation_coverage_no_union_bound))



    image_save_dir = "images/forPaper/validationPlots/"
    ## plot validation errs and prediction regions
    for i in range(len(all_x_valid[0])):

        errs_x = [all_x_l_valid[j][i] - all_x_valid[j][i] for j in range(len(all_x_valid))]
        errs_y = [all_y_l_valid[j][i] - all_y_valid[j][i] for j in range(len(all_y_valid))]

        cp_region_rad = D_cp/alphas[i]
        # cp_region_rad_min_area = D_cp_min_area/alphas_min_area[i]

        plt.clf()
        plt.scatter(errs_x,errs_y,label="Prediction Errors")
        plt.xlabel("X Error",fontsize="16")
        plt.ylabel("Y Error",fontsize="16")
        plt.axis('equal')
        plot_circle(0,0,cp_region_rad,'g-',label="Our approach")
        # plot_circle(0,0,cp_region_rad_min_area,'y-',label="min area")
        plot_circle(0,0,Ds_cp_union_bound[i],'r-',label="Union bound")
        # plot_circle(0,0,Ds_cp_no_union_bound[i],'k-',label="No union bound")
        plt.legend(fontsize="16",loc="upper left")
        ax=plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=12)

        plt.savefig(image_save_dir + "step" + str(i) + ".png")

        plt.clf()



    if PLOT_VALIDATION_TRACES:

        # trace_save_dir = "images/exp1/"
        trace_save_dir = "images/forPaper/"
        for j in range(len(all_x_valid)):

            if j >= NUM_VALID_TO_PLOT:
                continue

            x_vals = all_x_valid[j]
            y_vals = all_y_valid[j]

            x_hats = all_x_l_valid[j]
            y_hats = all_y_l_valid[j]

            plt.clf()

            # plot positions
            plt.plot(x_vals,y_vals,'*',color='cyan',label="Actual locations")

            # plot predictions
            plt.plot(x_hats,y_hats,'b*',label="Predicted locations")

            # plot conformal regions
            for t in range(p_len):

                if t==0:
                    leg_label_ours = "Our approach"
                    # leg_label_ours_min_area = "Min area"
                    leg_label_union = "Union bound"
                    leg_label_no = "No union bound"
                else:
                    leg_label_ours = "_nolegend_"
                    # leg_label_ours_min_area = "_nolegend_"
                    leg_label_union = "_nolegend_"
                    leg_label_no = "_nolegend_"
                # our cp region
                cp_region_rad = D_cp/alphas[t]
                plot_circle(x_hats[t],y_hats[t],cp_region_rad,'g-',label=leg_label_ours)

                # min area
                # cp_region_rad_min_area = D_cp_min_area/alphas_min_area[t]
                # plot_circle(x_hats[t],y_hats[t],cp_region_rad_min_area,'g-',label=leg_label_ours_min_area)

                # union bound cp region
                plot_circle(x_hats[t],y_hats[t],Ds_cp_union_bound[t],'r-',label=leg_label_union)

                # no union bound cp region
                # plot_circle(x_hats[t],y_hats[t],Ds_cp_no_union_bound[t],'k-',label=leg_label_no)
            
            plt.axis("equal")
            plt.legend(fontsize="14")
            ax=plt.gca()
            ax.tick_params(axis='both', which='major', labelsize=12)
            plt.savefig(trace_save_dir + "traces/trace" + str(j) + ".png")

            plt.clf()


        ## plot a few traces from the validation set, along with the predictions and preidiction regions for all 



    # for i in range(n):
    #     for t in range(T):
    #         fake_R_vals[i][t] = (t+1)*fake_R_vals[i][t]
    

    # optimzeTimeAlphasMinArgminHeuristic(fake_R_vals,delta,M)

    # optimzeTimeAlphasKKT(fake_R_vals,delta,M)
