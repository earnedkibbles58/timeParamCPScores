
import ndjson
import pandas as pd
import os
import math
import numpy as np
from numpy import linalg as LA
import time
import pickle
import random

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



from runParamSweepMatrixIneq import plot_circle


def computeDValsCircleTimeVaryingLinear(x_vals,y_vals,x_hats,y_hats,t_scalar,delta):

    # R_vals = [ math.sqrt((x_vals[i]-x_hats[i])**2 + (y_vals[i]-y_hats[i])**2) for i in range(len(x_vals))]
    R_vals = [max( [(1/((j+1)*t_scalar))*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))] ) for i in range(len(x_vals))]



    # [max([math.sqrt((1/j*t_scalar) (x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))]) for in range(len(x_vals))]

    # print(R_vals)


    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return(R_vals[ind_to_ret])

def computeDValsCircleTimeVaryingLinearNew(x_vals,y_vals,x_hats,y_hats,t_scalar,delta):


    ## compute d_val 

    # R_vals = [ math.sqrt((x_vals[i]-x_hats[i])**2 + (y_vals[i]-y_hats[i])**2) for i in range(len(x_vals))]
    R_vals = [max( [(1/((j+1)*t_scalar))*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))] ) for i in range(len(x_vals))]



    # [max([math.sqrt((1/j*t_scalar) (x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))]) for in range(len(x_vals))]

    # print(R_vals)


    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return(R_vals[ind_to_ret])


def computeDValsCircleTimeVaryingExponential(x_vals,y_vals,x_hats,y_hats,t_scalar,delta):

    # R_vals = [ math.sqrt((x_vals[i]-x_hats[i])**2 + (y_vals[i]-y_hats[i])**2) for i in range(len(x_vals))]
    R_vals = [max( [(1/((j+1)**t_scalar))*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))] ) for i in range(len(x_vals))]



    # [max([math.sqrt((1/j*t_scalar) (x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))]) for in range(len(x_vals))]

    # print(R_vals)


    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return(R_vals[ind_to_ret])


def computeDValsCircle(x_vals,y_vals,x_hats,y_hats,delta):

    R_vals = [ math.sqrt((x_vals[i]-x_hats[i])**2 + (y_vals[i]-y_hats[i])**2) for i in range(len(x_vals))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return(R_vals[ind_to_ret])


def checkCoverageCircleTimeVaryingLinear(x_vals,y_vals,x_hats,y_hats,t_scalar,d_val):

    # overall coverage
    R_vals = [max( [(1/((j+1)*t_scalar))*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))] ) for i in range(len(x_vals))]
    num_points_within = sum(r <= d_val for r in R_vals)
    coverage_pct_overall = float(num_points_within)/len(R_vals)
    
    coverage_pct_per_time = []
    for j in range(len(x_vals[0])):
        # per time step
        R_vals = [(1/((j+1)*t_scalar))*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for i in range(len(x_vals))]
        num_points_within = sum(r <= d_val for r in R_vals)
        coverage_pct_per_time.append(float(num_points_within)/len(R_vals))

    return coverage_pct_overall, coverage_pct_per_time


def checkCoverageCircleTimeVaryingExponential(x_vals,y_vals,x_hats,y_hats,t_scalar,d_val):

    # overall coverage
    R_vals = [max( [(1/((j+1)**t_scalar))*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for j in range(len(x_vals[i]))] ) for i in range(len(x_vals))]
    num_points_within = sum(r <= d_val for r in R_vals)
    coverage_pct_overall = float(num_points_within)/len(R_vals)
    
    coverage_pct_per_time = []
    for j in range(len(x_vals[0])):
        # per time step
        R_vals = [(1/((j+1)**t_scalar))*math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) for i in range(len(x_vals))]
        num_points_within = sum(r <= d_val for r in R_vals)
        coverage_pct_per_time.append(float(num_points_within)/len(R_vals))

    return coverage_pct_overall, coverage_pct_per_time




## TODO: write methods to check coverage for circle, simple ellipse, matrix ellipse and rectangle
def checkCoverageCircle(x_vals,y_vals,x_hats,y_hats,d_val):
    R_vals = [ math.sqrt((x_vals[i]-x_hats[i])**2 + (y_vals[i]-y_hats[i])**2) for i in range(len(x_vals))]

    num_points_within = sum(r <= d_val for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct


def checkCoverageCircleAllTimesAtOnce(x_vals,y_vals,x_hats,y_hats,d_vals):

    coverage_count = 0
    for i in range(len(x_vals)):

        temp = sum([1 if math.sqrt((x_vals[i][j]-x_hats[i][j])**2 + (y_vals[i][j]-y_hats[i][j])**2) > d_vals[j] else 0 for j in range(len(x_vals[i]))])
        if temp == 0:
            coverage_count += 1
        

    coverage_pct = float(coverage_count)/len(x_vals)
    return coverage_pct



save_path = "/data2/mcleav/conformalRNNs/parameterizedScores/code/firstExample/dvalues/"
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


print(d_value)


p_len = 20
delta = 0.05
calib_percent = 0.5
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

areas_circle = []
validation_coverage_circle = []

    

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

print(calib_ind)

all_x_calib = all_x[0:calib_ind]
all_y_calib = all_y[0:calib_ind]
all_x_l_calib = all_x_l[0:calib_ind]
all_y_l_calib = all_y_l[0:calib_ind]

all_x_valid = all_x[calib_ind:]
all_y_valid = all_y[calib_ind:]
all_x_l_valid = all_x_l[calib_ind:]
all_y_l_valid = all_y_l[calib_ind:]


print("Num data calib: " + str(len(all_x_calib)))
print("Num data valid: " + str(len(all_x_valid)))




# compute conformal regions using vanilla cp and circle non-conformity scores

d_vals_circles = []
d_vals_circles_union = []

coverages_circles = []
coverages_circles_union = []
for ind in range(p_len):

    x_temp = [all_x_calib[i][ind] for i in range(len(all_x_calib))]
    y_temp = [all_y_calib[i][ind] for i in range(len(all_x_calib))]
    x_l_temp = [all_x_l_calib[i][ind] for i in range(len(all_x_calib))]
    y_l_temp = [all_y_l_calib[i][ind] for i in range(len(all_x_calib))]

    d_val_temp = computeDValsCircle(x_temp,y_temp,x_l_temp,y_l_temp,delta)
    d_val_union_temp = computeDValsCircle(x_temp,y_temp,x_l_temp,y_l_temp,delta/p_len)

    d_vals_circles.append(d_val_temp)
    d_vals_circles_union.append(d_val_union_temp)

    x_temp_valid = [all_x_valid[i][ind] for i in range(len(all_x_valid))]
    y_temp_valid = [all_y_valid[i][ind] for i in range(len(all_x_valid))]
    x_l_temp_valid = [all_x_l_valid[i][ind] for i in range(len(all_x_valid))]
    y_l_temp_valid = [all_y_l_valid[i][ind] for i in range(len(all_x_valid))]


    coverage_circle = checkCoverageCircle(x_temp_valid,y_temp_valid,x_l_temp_valid,y_l_temp_valid,d_val_temp)
    coverage_circle_union = checkCoverageCircle(x_temp_valid,y_temp_valid,x_l_temp_valid,y_l_temp_valid,d_val_union_temp)

    coverages_circles.append(coverage_circle)
    coverages_circles_union.append(coverage_circle_union)

print("d values circle: " + str(d_vals_circles))
print("d values circle union bound: " + str(d_vals_circles_union))

time_scalars = np.logspace(-100,2,100)

d_vals_linear = [computeDValsCircleTimeVaryingLinear(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,time_scalar,delta) for time_scalar in time_scalars]
total_coverages_circle_linear_time = [sum(checkCoverageCircleTimeVaryingLinear(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,time_scalars[i],d_vals_linear[i])[1])/p_len for i in range(len(time_scalars))]

d_vals_exp = [computeDValsCircleTimeVaryingExponential(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,time_scalar,delta) for time_scalar in time_scalars]
total_coverages_circle_exp_time = [sum(checkCoverageCircleTimeVaryingExponential(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,time_scalars[i],d_vals_linear[i])[1])/p_len for i in range(len(time_scalars))]


time_scalar_opt_linear = time_scalars[total_coverages_circle_linear_time.index(min(total_coverages_circle_linear_time))]
d_val_opt_linear = d_vals_linear[total_coverages_circle_linear_time.index(min(total_coverages_circle_linear_time))]
average_coverage_linear = total_coverages_circle_linear_time[total_coverages_circle_linear_time.index(min(total_coverages_circle_linear_time))]
d_vals_opt_linear_over_time = [d_val_opt_linear*(ind+1)*time_scalar_opt_linear for ind in range(p_len) ]

print("Optimal time scalar: " + str(time_scalar_opt_linear))
print("d val: " + str(d_val_opt_linear))
print("Average coverage over time: " + str(average_coverage_linear))
print("d vals over time: " + str(d_vals_opt_linear_over_time[0:5]))



## print stuff for time scalar = 1
time_scalar_opt_linear_1 = 1
d_val_opt_linear_1 = computeDValsCircleTimeVaryingLinear(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,time_scalar_opt_linear_1,delta)
average_coverage_linear_1 = sum(checkCoverageCircleTimeVaryingLinear(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,time_scalar_opt_linear_1,d_val_opt_linear_1)[1])/p_len
d_vals_opt_linear_over_time_1 = [d_val_opt_linear_1*(ind+1)*time_scalar_opt_linear_1 for ind in range(p_len) ]

print("Optimal time scalar: " + str(time_scalar_opt_linear_1))
print("d val: " + str(d_val_opt_linear_1))
print("Average coverage over time: " + str(average_coverage_linear_1))
print("d vals over time: " + str(d_vals_opt_linear_over_time_1[0:5]))





# print(total_coverages_circle_linear_time)


    

# print("d value circle: " + str(d_val_circle))


# ## print coverage
# coverage_circle_calib_overall,coverage_circle_calib_per_time = checkCoverageCircleTimeVaryingLinear(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,time_scalar,d_val_circle)
# coverage_circle_valid_overall,coverage_circle_valid_per_time = checkCoverageCircleTimeVaryingLinear(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,time_scalar,d_val_circle)

# print("Overall calibration coverage for circle: " + str(coverage_circle_calib_overall))
# print("Overall validation coverage for circle: " + str(coverage_circle_valid_overall))


# print("Calibration coverage over time for circle: " + str(coverage_circle_calib_per_time))
# print("Validation coverage over time for circle: " + str(coverage_circle_valid_per_time))



## compute coverage for each conformal region
coverage_circle = checkCoverageCircleAllTimesAtOnce(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,d_vals_circles)
coverage_circle_union = checkCoverageCircleAllTimesAtOnce(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,d_vals_circles_union)
coverage_linear_time = checkCoverageCircleAllTimesAtOnce(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,d_vals_opt_linear_over_time)
coverage_linear_time_1 = checkCoverageCircleAllTimesAtOnce(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,d_vals_opt_linear_over_time_1)

print("Trace wise coverages")
print("Circle: " + str(coverage_circle))
print("Circle union: " + str(coverage_circle_union))
print("Linear time: " + str(coverage_linear_time))
print("Linear time: " + str(coverage_linear_time_1))



## plot conformal regions

for ind in range(p_len):

    err_x = [all_x_l_valid[i][ind]-all_x_valid[i][ind] for i in range(len(all_x_valid))]
    err_y = [all_y_l_valid[i][ind]-all_y_valid[i][ind] for i in range(len(all_y_valid))]

    plt.scatter(err_x,err_y)
    plt.xlabel("X Error")
    plt.ylabel("Y Error")
    plt.axis('equal')

    plot_circle(0, 0, d_vals_opt_linear_over_time[ind], color="-g") ## TODO: does the (j+1)*time_scalar need to be squared or square rooted?
    plot_circle(0, 0, d_vals_opt_linear_over_time_1[ind], color="-b") ## TODO: does the (j+1)*time_scalar need to be squared or square rooted?
    plot_circle(0,0,d_vals_circles[ind], color = '-k')
    plot_circle(0,0,d_vals_circles_union[ind], color = '-r')
    plt.savefig("images/timeVaryLinear/errPlots/timeHorizon" + str(ind) + ".png")
    plt.clf()



plt.plot(d_vals_opt_linear_over_time, 'g*',label="linear time varying")
plt.plot(d_vals_opt_linear_over_time_1, 'b*',label="linear time  1")
plt.plot(d_vals_circles, 'k*',label="circle")
plt.plot(d_vals_circles_union, 'r*', label="circle (union bound)")
plt.legend()
plt.xlabel("Prediction Time Horizon")
plt.ylabel("Conformal Region Radius")
plt.savefig("images/timeVaryLinear/dvalRadii.png")

