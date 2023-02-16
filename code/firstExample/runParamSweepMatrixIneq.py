
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


def computeEllipseArea(a_1,a_2,b):
    a_major = (1/(4*a_2**2-4*a_1)) * (-math.sqrt( 2*(( (4*a_2**2-4*a_1) * -b)) * ( (1+a_1) + math.sqrt( (1-a_1)**2 + 4*(a_2)**2 ))))
    a_minor = (1/(4*a_2**2-4*a_1)) * (-math.sqrt( 2*(( (4*a_2**2-4*a_1) * -b)) * ( (1+a_1) - math.sqrt( (1-a_1)**2 + 4*(a_2)**2 ))))

    assert a_major > 0
    assert a_minor > 0
    assert a_minor >= a_minor

    area = math.pi*a_major*a_minor

    return area
    



def computeMajorMinorRotGeneralEllipse(a_1,a_2,b):
    a_major = (1/(4*a_2**2-4*a_1)) * (-math.sqrt( 2*(( (4*a_2**2-4*a_1) * -b)) * ( (1+a_1) + math.sqrt( (1-a_1)**2 + 4*(a_2)**2 ))))
    a_minor = (1/(4*a_2**2-4*a_1)) * (-math.sqrt( 2*(( (4*a_2**2-4*a_1) * -b)) * ( (1+a_1) - math.sqrt( (1-a_1)**2 + 4*(a_2)**2 ))))

    #3 compute rotation angle
    # equation 

    if a_2 == 0 and a_1 >= 1:
        rot = 0
    elif a_2 == 0 and a_1 < 1:
        rot = math.pi/2
    else:
        rot = math.atan((1/(2*a_2))*(a_1-1-math.sqrt((1-a_1)**2+4*a_2**2)))

    return a_major,a_minor,rot


## x^T Q x <= b
# Q = [1 a1; a1 a2] > 0
def computeDValsMatrixEllipse(x_vals,y_vals,x_hats,y_hats,a1,a2,delta):

    assert a1 >0
    assert a1 > a2**2

    R_vals = [ (x_vals[i]-x_hats[i])**2 + 2*a2*(x_vals[i]-x_hats[i])*(y_vals[i]-y_hats[i]) + a1*(y_vals[i]-y_hats[i])**2 for i in range(len(x_vals))]

    R_vals.sort()

    assert R_vals[0] >= 0

    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return(R_vals[ind_to_ret])




def computeDValsEllipse(x_vals,y_vals,x_hats,y_hats,a,delta):

    R_vals = [ (x_vals[i]-x_hats[i])**2 + a*(y_vals[i]-y_hats[i])**2 for i in range(len(x_vals))]

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


def computeDValsRectangle(x_vals,y_vals,x_hats,y_hats,a,delta):

    R_vals = [ max(abs(x_vals[i]-x_hats[i]), a*abs(y_vals[i]-y_hats[i])) for i in range(len(x_vals))]

    R_vals.sort()
    R_vals.append(max(R_vals))

    ind_to_ret = math.ceil(len(R_vals)*(1-delta))
    return(R_vals[ind_to_ret])
    


## TODO: write methods to check coverage for circle, simple ellipse, matrix ellipse and rectangle
def checkCoverageCircle(x_vals,y_vals,x_hats,y_hats,d_val):
    R_vals = [ math.sqrt((x_vals[i]-x_hats[i])**2 + (y_vals[i]-y_hats[i])**2) for i in range(len(x_vals))]

    num_points_within = sum(r <= d_val for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct


## TODO: write methods to check coverage for circle, simple ellipse, matrix ellipse and rectangle
def checkCoverageEllipse(x_vals,y_vals,x_hats,y_hats,a,d_val):
    R_vals = [ (x_vals[i]-x_hats[i])**2 + a*(y_vals[i]-y_hats[i])**2 for i in range(len(x_vals))]

    num_points_within = sum(r <= d_val for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct

## TODO: write methods to check coverage for circle, simple ellipse, matrix ellipse and rectangle
def checkCoverageRectangle(x_vals,y_vals,x_hats,y_hats,a,d_val):
    R_vals = [ max(abs(x_vals[i]-x_hats[i]), a*abs(y_vals[i]-y_hats[i])) for i in range(len(x_vals))]

    num_points_within = sum(r <= d_val for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct

def checkCoverageMatrixEllipse(x_vals,y_vals,x_hats,y_hats,a1,a2,d_val):

    assert a1 >0
    assert a1 > a2**2

    R_vals = [ (x_vals[i]-x_hats[i])**2 + 2*a2*(x_vals[i]-x_hats[i])*(y_vals[i]-y_hats[i]) + a1*(y_vals[i]-y_hats[i])**2 for i in range(len(x_vals))]

    num_points_within = sum(r <= d_val for r in R_vals)
    coverage_pct = float(num_points_within)/len(R_vals)
    return coverage_pct


def plotEllipse(a_x,a_y,rot,x_pos,y_pos,color="-b"):

    u=x_pos       #x-position of the center
    v=y_pos      #y-position of the center
    a=a_x       #radius on the x-axis
    b=a_y      #radius on the y-axis
    t_rot=rot #rotation angle

    t = np.linspace(0, 2*math.pi, 100)
    Ell = np.array([a*np.cos(t) , b*np.sin(t)])  
        #u,v removed to keep the same center location
    R_rot = np.array([[math.cos(t_rot) , -math.sin(t_rot)],[math.sin(t_rot) , math.cos(t_rot)]])  
        #2-D rotation matrix

    Ell_rot = np.zeros((2,Ell.shape[1]))
    for i in range(Ell.shape[1]):
        Ell_rot[:,i] = np.dot(R_rot,Ell[:,i])

    # plt.plot( u+Ell[0,:] , v+Ell[1,:] )     #initial ellipse
    plt.plot( u+Ell_rot[0,:] , v+Ell_rot[1,:],color )    #rotated ellipse



def plot_circle(x, y, size, color="-b"):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)





def main():


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


    ind = 0

    areas_circle = []
    min_areas_simple_ellipse = []
    min_areas_rectangle = []
    min_areas_matrix_ellipse = []


    validation_coverage_circle = []
    validation_coverage_simple_ellipse = []
    validation_coverage_rectangle = []
    validation_coverage_matrix_ellipse = []

    for ind in range(p_len):

        print("ind: " + str(ind))
        

        all_x = []
        all_y = []
        all_x_l = []
        all_y_l = []

        all_x_err = []
        all_y_err = []

        for k in keys_x:
            # x = x_value[dict_key_x][k]
            # y = y_value[dict_key_y][k]
            # x_l = x_l_value[dict_key_x_l][k]
            # y_l = y_l_value[dict_key_y_l][k]


            # print(x_value[dict_key_x][k])
            # print(y_value[dict_key_y][k])
            # print(x_l_value[dict_key_x_l][k])
            # print(y_l_value[dict_key_y_l][k ])


            final_key = list(x_value[dict_key_x][k].keys())[0]

            x = x_value[dict_key_x][k][final_key]
            y = y_value[dict_key_y][k][final_key]
            x_l = x_l_value[dict_key_x_l][k][final_key]
            y_l = y_l_value[dict_key_y_l][k][final_key]



            # print(len(x))
            # print(len(y))
            # print(len(x_l))
            # print(len(y_l))

            all_x.append(x[p_len+ind])
            all_y.append(y[p_len+ind])
            all_x_l.append(x_l[0+ind])
            all_y_l.append(y_l[0+ind])

            all_x_err.append(x_l[0+ind]-x[p_len+ind])
            all_y_err.append(y_l[0+ind]-y[p_len+ind])


        ## create train/test split
        temp = list(zip(all_x, all_y, all_x_l,all_y_l))
        random.shuffle(temp)

        res1,res2,res3,res4 = zip(*temp)
        all_x,all_y,all_x_l,all_y_l = list(res1),list(res2),list(res3),list(res4)

        calib_ind = round(len(all_x)*calib_percent)
        
        all_x_calib = all_x[0:calib_ind]
        all_y_calib = all_y[0:calib_ind]
        all_x_l_calib = all_x_l[0:calib_ind]
        all_y_l_calib = all_y_l[0:calib_ind]

        all_x_valid = all_x[calib_ind:]
        all_y_valid = all_y[calib_ind:]
        all_x_l_valid = all_x_l[calib_ind:]
        all_y_l_valid = all_y_l[calib_ind:]


        print("Num data: " + str(len(all_x_calib)))

        a_values = np.logspace(-2,2,100) # for non rotated ellipse and rectangle
        a_1_values = np.logspace(-2,2,100) # for matrix representation
        a_2_values = np.linspace(-1,1,100)

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


                ## TODO: figure out the area of the ellipse based on a_1, a_2, d_val
                area = computeEllipseArea(a_1,a_2,d_val)
                areas_general_ellipse.append(area)




        # plt.plot(a_values,areas_ellipse)
        # plt.xlabel("a value")
        # plt.ylabel("conformal region area")
        # plt.savefig("images/ellipse/firstFig_" + str(ind) + ".png")

        # plt.xscale('log')
        # plt.savefig("images/ellipse/firstFigLogX_" + str(ind) + ".png")
        # plt.clf()


        # plt.plot(a_values,areas_rectangle)
        # plt.xlabel("a value")
        # plt.ylabel("conformal region area")
        # plt.savefig("images/rectangle/firstFig_" + str(ind) + ".png")

        # plt.xscale('log')
        # plt.savefig("images/rectangle/firstFigLogX_" + str(ind) + ".png")
        # plt.clf()


        ax = plt.axes(projection='3d')
        ax.scatter3D(all_a_1, all_a_2, areas_general_ellipse, c=areas_general_ellipse, cmap='Greens')

        if makePlots:
            ax.set_xlabel("a_1 values")
            ax.set_ylabel("a_2 values")
            ax.set_zlabel("conformal region area")
            plt.savefig("images/generalEllipse/firstFig_" + str(ind) + ".png")
            ax.set_xscale("log")
            ax.set_yscale("log")
            plt.savefig("images/generalEllipse/firstFigLogX_" + str(ind) + ".png")
            plt.clf()


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

        validation_coverage_circle.append(coverage_circle_valid)
        validation_coverage_simple_ellipse.append(coverage_ellipse_valid)
        validation_coverage_rectangle.append(coverage_rectangle_valid)
        validation_coverage_matrix_ellipse.append(coverage_matrix_ellipse_valid)


        if makePlots:
            plt.clf()
            plt.scatter(all_x_err,all_y_err)
            plt.xlabel("X Error")
            plt.ylabel("Y Error")
            plt.axis('equal')

            # plot circle
            plot_circle(0, 0, d_val_circle, color="-k")

            # plot rectangle
            plt.gca().add_patch(Rectangle((-min_d_rectangle,-min_d_rectangle/min_a_rectangle),2*min_d_rectangle,2*min_d_rectangle/min_a_rectangle,linewidth=1,edgecolor='b',facecolor='none'))

            # plot simple ellipse
            plotEllipse(math.sqrt(min_d_ellipse),math.sqrt(min_d_ellipse/min_a_ellipse),0,0,0,color="-r")

            # plot matrix ellipse
            a_maj_gen,a_min_gen,a_rot_gen = computeMajorMinorRotGeneralEllipse(min_a_general_ellipse[0],min_a_general_ellipse[1],min_d_general_ellipse)
            plotEllipse(a_maj_gen,a_min_gen,a_rot_gen,0,0,color="-g")

            plt.savefig("images/errScatterPlots/predHorizon" + str(ind) + ".png")
            plt.clf()

        

    if makePlots:
        plt.clf()
        plt.plot(areas_circle,'k*',label="circle")
        plt.plot(min_areas_simple_ellipse,'r*',label="simple ellipse")
        plt.plot(min_areas_rectangle,'b*',label="simple rectangle")
        plt.plot(min_areas_matrix_ellipse,'g*',label="matrix ellipse")
        plt.xlabel('Prediction Horizon')
        plt.ylabel('Conformal Region Area')
        plt.legend()
        plt.savefig("images/conformalRegionAreas.png")
        plt.clf()



        plt.plot(validation_coverage_circle,'k*',label="circle")
        plt.plot(validation_coverage_simple_ellipse,'r*',label="simple ellipse")
        plt.plot(validation_coverage_rectangle,'b*',label="simple rectangle")
        plt.plot(validation_coverage_matrix_ellipse,'g*',label="matrix ellipse")
        plt.axhline(y=1-delta, color='k', linestyle='--')

        plt.xlabel('Prediction Horizon')
        plt.ylabel('Conformal Region Coverage on Validation Data')
        plt.legend()
        plt.savefig("images/conformalRegionValidationCoverage.png")
        plt.clf()









if __name__ == "__main__":
    main()




