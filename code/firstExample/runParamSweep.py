
import ndjson
import pandas as pd
import os
import math
import numpy as np
from numpy import linalg as LA
import time
import pickle
import matplotlib.pyplot as plt



def computeDValsEllipse(x_vals,y_vals,x_hats,y_hats,a,delta):

    R_vals = [ (x_vals[i]-x_hats[i])**2 + a*(y_vals[i]-y_hats[i])**2 for i in range(len(x_vals))]

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

for ind in range(p_len):

    print("ind: " + str(ind))

    all_x = []
    all_y = []
    all_x_l = []
    all_y_l = []


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


        # print(all_x)
        # print(all_y)
        # print(all_x_l)
        # print(all_y_l)





    a_values = np.logspace(-2,2,100)

    d_vals_ellipse = [computeDValsEllipse(all_x,all_y,all_x_l,all_y_l,a,delta) for a in a_values]
    areas_ellipse = [math.pi*d_vals_ellipse[i]/math.sqrt(a_values[i]) for i in range(len(a_values))]

    d_vals_rectangle = [computeDValsRectangle(all_x,all_y,all_x_l,all_y_l,a,delta) for a in a_values]
    areas_rectangle = [d_vals_rectangle[i]**2/a_values[i] for i in range(len(a_values))]


    plt.plot(a_values,areas_ellipse)
    plt.xlabel("a value")
    plt.ylabel("conformal region area")
    plt.savefig("images/ellipse/firstFig_" + str(ind) + ".png")

    plt.xscale('log')
    plt.savefig("images/ellipse/firstFigLogX_" + str(ind) + ".png")
    plt.clf()


    plt.plot(a_values,areas_rectangle)
    plt.xlabel("a value")
    plt.ylabel("conformal region area")
    plt.savefig("images/rectangle/firstFig_" + str(ind) + ".png")

    plt.xscale('log')
    plt.savefig("images/rectangle/firstFigLogX_" + str(ind) + ".png")
    plt.clf()



    min_a_ellipse = a_values[areas_ellipse.index(min(areas_ellipse))]
    print("min a ellipse: " + str(min_a_ellipse))
    print("min area ellipse: " + str(min(areas_ellipse)))

    min_a_rectangle = a_values[areas_rectangle.index(min(areas_rectangle))]
    print("min a rectangle: " + str(min_a_rectangle))
    print("min area rectangle: " + str(min(areas_rectangle)))

