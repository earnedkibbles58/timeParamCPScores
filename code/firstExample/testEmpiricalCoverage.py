
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

from runParamSweepMatrixIneq import *


def computeAverageCoveragePerTime(coverageAllTimes):

    averages = []
    for i in range(len(coverageAllTimes[0])):

        coverages = [c[i] for c in coverageAllTimes]
        averages.append(np.mean(coverages))
    return averages












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
calib_percent = 0.25
valid_percent = 0.25
makePlots = True
numTrials = 100


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


ind = 0


validation_coverage_circle = []
validation_coverage_simple_ellipse = []
validation_coverage_rectangle = []
validation_coverage_matrix_ellipse = []


for trial in range(numTrials):

    print("Trial " + str(trial),flush=True)

    validation_coverage_circle_temp = []
    validation_coverage_simple_ellipse_temp = []
    validation_coverage_rectangle_temp = []
    validation_coverage_matrix_ellipse_temp = []

    for ind in range(p_len):

        print("ind: " + str(ind))
        

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


            all_x.append(x[p_len+ind])
            all_y.append(y[p_len+ind])
            all_x_l.append(x_l[0+ind])
            all_y_l.append(y_l[0+ind])



        # draw calibration data
        temp1 = list(zip(all_x, all_y, all_x_l,all_y_l))
        random.shuffle(temp1)
        res1,res2,res3,res4 = zip(*temp1)
        all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib = list(res1),list(res2),list(res3),list(res4)
        
        calib_ind = round(len(all_x)*calib_percent)
        
        all_x_calib = all_x_calib[0:calib_ind]
        all_y_calib = all_y_calib[0:calib_ind]
        all_x_l_calib = all_x_l_calib[0:calib_ind]
        all_y_l_calib = all_y_l_calib[0:calib_ind]



        # draw validation data
        temp2 = list(zip(all_x, all_y, all_x_l,all_y_l))
        random.shuffle(temp2)
        res1,res2,res3,res4 = zip(*temp2)
        all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid = list(res1),list(res2),list(res3),list(res4)

        valid_ind = round(len(all_x)*valid_percent)

        all_x_valid = all_x_valid[0:valid_ind]
        all_y_valid = all_y_valid[0:valid_ind]
        all_x_l_valid = all_x_l_valid[0:valid_ind]
        all_y_l_valid = all_y_l_valid[0:valid_ind]


        print("Num data calib: " + str(len(all_x_calib)))
        print("Num data valid: " + str(len(all_x_valid)))

        a_values = np.logspace(-2,2,50) # for non rotated ellipse and rectangle
        a_1_values = np.logspace(-2,2,50) # for matrix representation
        a_2_values = np.linspace(-1,1,50)

        d_val_circle = computeDValsCircle(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,delta)
        area_circle = math.pi*d_val_circle**2

        d_vals_ellipse = [computeDValsEllipse(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,a,delta) for a in a_values]
        areas_ellipse = [math.pi*d_vals_ellipse[i]/math.sqrt(a_values[i]) for i in range(len(a_values))]

        d_vals_rectangle = [computeDValsRectangle(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,a,delta) for a in a_values]
        areas_rectangle = [4*d_vals_rectangle[i]**2/a_values[i] for i in range(len(a_values))]


        all_a_1 = []
        all_a_2 = []
        d_vals_matrix = []
        areas_general_ellipse = []
        for a_1 in a_1_values:
            for a_2 in a_2_values:

                if a_1 < a_2**2:
                    continue

                d_val = computeDValsMatrixEllipse(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,a_1,a_2,delta)
                
                all_a_1.append(a_1)
                all_a_2.append(a_2)
                d_vals_matrix.append(d_val)


                area = computeEllipseArea(a_1,a_2,d_val)
                areas_general_ellipse.append(area)

        print("min area circle: " + str(area_circle))

        min_a_ellipse = a_values[areas_ellipse.index(min(areas_ellipse))]
        min_d_ellipse = d_vals_ellipse[areas_ellipse.index(min(areas_ellipse))]

        print("min a ellipse: " + str(min_a_ellipse))
        print("min area ellipse: " + str(min(areas_ellipse)))

        min_a_rectangle = a_values[areas_rectangle.index(min(areas_rectangle))]
        min_d_rectangle = d_vals_rectangle[areas_rectangle.index(min(areas_rectangle))]
        print("min a rectangle: " + str(min_a_rectangle))
        print("min area rectangle: " + str(min(areas_rectangle)))

        min_a_general_ellipse = [all_a_1[areas_general_ellipse.index(min(areas_general_ellipse))],all_a_2[areas_general_ellipse.index(min(areas_general_ellipse))]]
        min_d_general_ellipse = d_vals_matrix[areas_general_ellipse.index(min(areas_general_ellipse))]
        print("min a general ellipse: " + str(min_a_general_ellipse))
        print("min area general ellipse: " + str(min(areas_general_ellipse)))

        areas_circle.append(area_circle)
        min_areas_simple_ellipse.append(min(areas_ellipse))
        min_areas_rectangle.append(min(areas_rectangle))
        min_areas_matrix_ellipse.append(min(areas_general_ellipse))



        ## print coverage
        coverage_circle_calib = checkCoverageCircle(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,d_val_circle)
        coverage_circle_valid = checkCoverageCircle(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,d_val_circle)
        print("Calibration coverage for circle: " + str(coverage_circle_calib))
        print("Validation coverage for circle: " + str(coverage_circle_valid))

        coverage_ellipse_calib = checkCoverageEllipse(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,min_a_ellipse,min_d_ellipse)
        coverage_ellipse_valid = checkCoverageEllipse(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,min_a_ellipse,min_d_ellipse)
        print("Calibration coverage for ellipse: " + str(coverage_ellipse_calib))
        print("Validation coverage for ellipse: " + str(coverage_ellipse_valid))

        coverage_rectangle_calib = checkCoverageRectangle(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,min_a_rectangle,min_d_rectangle)
        coverage_rectangle_valid = checkCoverageRectangle(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,min_a_rectangle,min_d_rectangle)
        print("Calibration coverage for rectangle: " + str(coverage_rectangle_calib))
        print("Validation coverage for rectangle: " + str(coverage_rectangle_valid))

        coverage_matrix_ellipse_calib = checkCoverageMatrixEllipse(all_x_calib,all_y_calib,all_x_l_calib,all_y_l_calib,min_a_general_ellipse[0],min_a_general_ellipse[1],min_d_general_ellipse)
        coverage_matrix_ellipse_valid = checkCoverageMatrixEllipse(all_x_valid,all_y_valid,all_x_l_valid,all_y_l_valid,min_a_general_ellipse[0],min_a_general_ellipse[1],min_d_general_ellipse)
        print("Calibration coverage for matrix ellipse: " + str(coverage_matrix_ellipse_calib))
        print("Validation coverage for matrix ellipse: " + str(coverage_matrix_ellipse_valid))

        validation_coverage_circle_temp.append(coverage_circle_valid)
        validation_coverage_simple_ellipse_temp.append(coverage_ellipse_valid)
        validation_coverage_rectangle_temp.append(coverage_rectangle_valid)
        validation_coverage_matrix_ellipse_temp.append(coverage_matrix_ellipse_valid)


    validation_coverage_circle.append(validation_coverage_circle_temp)
    validation_coverage_simple_ellipse.append(validation_coverage_simple_ellipse_temp)
    validation_coverage_rectangle.append(validation_coverage_rectangle_temp)
    validation_coverage_matrix_ellipse.append(validation_coverage_matrix_ellipse_temp)


## get average coverage over time 

average_coverage_circle = computeAverageCoveragePerTime(validation_coverage_circle)
average_coverage_simple_ellipse = computeAverageCoveragePerTime(validation_coverage_simple_ellipse)
average_coverage_rectangle = computeAverageCoveragePerTime(validation_coverage_rectangle)
average_coverage_matrix_ellipse = computeAverageCoveragePerTime(validation_coverage_matrix_ellipse)

print("Average coverage circle: " + str(average_coverage_circle))
print("Average coverage simple ellipse: " + str(average_coverage_simple_ellipse))
print("Average coverage rectangle: " + str(average_coverage_rectangle))
print("Average coverage matrix ellipse: " + str(average_coverage_matrix_ellipse))


if makePlots:

    ## plot histograms of coverage

    # n_bins = 100
    for i in range(p_len):

        plt.clf()
        ## plot circle
        coverages = [c[i] for c in validation_coverage_circle]
        plt.hist(coverages)
        plt.axvline(x = 1-delta, color = 'k')
        plt.xlabel("Empirical Coverage")
        plt.ylabel("Counts")
        plt.savefig("images/coverageHists/circle/predHorizon" + str(i) + ".png")
        plt.clf()

        ## plot simple ellipse
        coverages = [c[i] for c in validation_coverage_simple_ellipse]
        plt.hist(coverages)
        plt.axvline(x = 1-delta, color = 'k--')
        plt.xlabel("Empirical Coverage")
        plt.ylabel("Counts")
        plt.savefig("images/coverageHists/simpleEllipse/predHorizon" + str(i) + ".png")
        plt.clf()

        ## plot rectangle
        coverages = [c[i] for c in validation_coverage_rectangle]
        plt.hist(coverages)
        plt.axvline(x = 1-delta, color = 'k--')
        plt.xlabel("Empirical Coverage")
        plt.ylabel("Counts")
        plt.savefig("images/coverageHists/rectangle/predHorizon" + str(i) + ".png")
        plt.clf()

        ## plot matrix ellipse
        coverages = [c[i] for c in validation_coverage_matrix_ellipse]
        plt.hist(coverages)
        plt.axvline(x = 1-delta, color = 'k--')
        plt.xlabel("Empirical Coverage")
        plt.ylabel("Counts")
        plt.savefig("images/coverageHists/matrixEllipse/predHorizon" + str(i) + ".png")
        plt.clf()

