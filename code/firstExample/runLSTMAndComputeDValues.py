# (Conformal Prediction) Computing d values

# Get d values

import ndjson
import pandas as pd
import os
import math
import numpy as np
from numpy import linalg as LA
import time

H = 20
o_len = 20
p_len = 20

priv = []
pred = []
pv = "test_private_"
pd = "test_pred_"

d = "d"
x_cha = "x"
y_cha = "y"
x_l = "x_LSTM"
y_l = "y_LSTM"

d_list = []
x_list = []
y_list = []
x_l_list = []
y_l_list = []

for i in range(p_len):
    priv.append(pv + str(o_len+i))
    pred.append(pd + str(o_len+i) + "/")
    d_list.append(d+str(i+o_len))
    x_list.append(x_cha+str(i+o_len))
    y_list.append(y_cha+str(i+o_len))
    x_l_list.append(x_l+str(i+o_len))
    y_l_list.append(y_l+str(i+o_len))




# Modify the pred_lstm and path
from sys import maxunicode

useMaxNorm = True

pred_lstm = 'lstm_goals_social_None_20_2000_modes1/'
# path = "/Users/joey/Trajnet_test/trajnetplusplusbaselines/DATA_BLOCK/synth_data/"
path = "/data2/mcleav/conformalRNNs/icra_2022/code/Trajnet_test/trajnetplusplusbaselines/DATA_BLOCK/synth_data/"

d_value = {}
x_value = {}
y_value = {}
x_l_value = {}
y_l_value = {}

start_time = time.time()

for q in range(p_len):
    os.chdir(path + str(priv[q]))
    with open('orca_three_synth.ndjson', 'r') as f:
        original_data=ndjson.load(f)    
    os.chdir(path + str(pred[q])+ str(pred_lstm))
    with open('orca_three_synth.ndjson', 'r') as f:
        predict_data=ndjson.load(f)
    
    d_value[d_list[q]] = []
    x_value[x_list[q]] = []
    y_value[y_list[q]] = []
    x_l_value[x_l_list[q]] = []
    y_l_value[y_l_list[q]] = []
    
    number_of_scenes = []
    p_values = []
    s_values = []
    e_values = []
    IDs = []
    # increase if there are more pedestrian
    
    peds = ["Ped 1"] #, "Ped 2", "Ped 3"]
    ID = "ID: "

    for i in range(len(original_data)):
        if list(original_data[i].keys())[0] == 'scene':
            number_of_scenes.append(list(original_data[i].values())[0].get("id"))

    for i in range(len(original_data)):
        if list(original_data[i].keys())[0] == 'track':
            p_values.append(list(original_data[i].values())[0].get("p"))

    p_value = []
    for i in p_values:
        if i not in p_value:
            p_value.append(i)
    
    for i in range(len(predict_data)):
        if list(predict_data[i].keys())[0] == 'scene':
            s_values.append(list(predict_data[i].values())[0].get("s"))
            e_values.append(list(predict_data[i].values())[0].get("e"))
        
    for i in range(len(number_of_scenes)):
        IDs.append(number_of_scenes[i])
        IDs[i] = ID + str(IDs[i])

    p_values_for_scene_id = {}
    for i in range(len(s_values)):
        p_values_for_scene_id[IDs[i]] = []
        p = []
        pp = []
        for j in range(len(predict_data)):
            if list(predict_data[j].keys())[0] == 'track':
                if list(predict_data[j].values())[0].get("scene_id") == number_of_scenes[i]:
                    p.append(list(predict_data[j].values())[0].get("p"))
        for k in p:
            if k not in pp:
                pp.append(k)
        p_values_for_scene_id.update({IDs[i] : pp})
    
    x_original_trajectories = {}
    y_original_trajectories = {}

    for i in range(len(s_values)):
        x_original_trajectories[IDs[i]] = {}
        y_original_trajectories[IDs[i]] = {}
        for t in range(len(peds)):
            x_original_trajectories[IDs[i]][peds[t]] = []
            y_original_trajectories[IDs[i]][peds[t]] = []   
            x_orig_traj = []
            y_orig_traj = []
            for j in range(len(original_data)):
                if list(original_data[j].keys())[0] == 'track':
                    if list(original_data[j].values())[0].get("p") == p_values_for_scene_id[IDs[i]][t]:
                        for k in range(s_values[i], e_values[i]+1):
                            if list(original_data[j].values())[0].get("f") == k:
                                x_orig_traj.append(list(original_data[j].values())[0].get("x"))
                                y_orig_traj.append(list(original_data[j].values())[0].get("y"))
            x_original_trajectories[IDs[i]].update({peds[t] : x_orig_traj})
            y_original_trajectories[IDs[i]].update({peds[t] : y_orig_traj})

    # Prediction Trajectories
    x_predict_trajectories = {}
    y_predict_trajectories = {}

    for i in range(len(s_values)):
        x_predict_trajectories[IDs[i]] = {}
        y_predict_trajectories[IDs[i]] = {}
        x_pred_traj = []
        y_pred_traj = []
        for j in range(len(predict_data)):
            if list(predict_data[j].keys())[0] == 'track':
                if list(predict_data[j].values())[0].get("scene_id") == number_of_scenes[i]:
                    x_pred_traj.append(list(predict_data[j].values())[0].get("x"))
                    y_pred_traj.append(list(predict_data[j].values())[0].get("y"))
        for t in range(len(peds)):
            x_predict_trajectories[IDs[i]][peds[t]] = []
            y_predict_trajectories[IDs[i]][peds[t]] = []
            x_predict_trajectories[IDs[i]].update({peds[t] : x_pred_traj[int(p_len)*t:int(p_len)*(t+1)]})
            y_predict_trajectories[IDs[i]].update({peds[t] : y_pred_traj[int(p_len)*t:int(p_len)*(t+1)]})

    find_th_number_x = []
    find_th_number_y = []

    for i in range(len(IDs)):
        for j in range(len(peds)):
            if not len(x_original_trajectories[IDs[i]][peds[j]]) == (q+int(o_len+p_len)):
                x_original_trajectories[IDs[i]][peds[j]] = [9999] * (q+int(o_len+p_len))
                find_th_number_x.append(np.array([i, j]))

    for i in range(len(IDs)):
        for j in range(len(peds)):
            if not len(y_original_trajectories[IDs[i]][peds[j]]) == (q+int(o_len+p_len)):
                y_original_trajectories[IDs[i]][peds[j]] = [9999] * (q+int(o_len+p_len))
                find_th_number_y.append(np.array([i, j]))
              
    for i in range(len(find_th_number_x)):
        x_predict_trajectories[IDs[find_th_number_x[i][0]]][peds[find_th_number_x[i][1]]] = [0] * 20

    for i in range(len(find_th_number_y)):
        y_predict_trajectories[IDs[find_th_number_y[i][0]]][peds[find_th_number_y[i][1]]] = [0] * 20

    R_value_x = {}
    R_value_y = {}
    R_value = {}

    list_R_value_x = []
    list_R_value_y = []
    list_R_value = []
    for i in range(len(IDs)):
        R_value_x[IDs[i]] = {}
        R_value_y[IDs[i]] = {}
        R_value[IDs[i]] = {}
        for j in range(len(peds)):
            R_value_x[IDs[i]][peds[j]] = [] 
            R_value_y[IDs[i]][peds[j]] = []
            R_value[IDs[i]][peds[j]] = []
            list_R_value_x = []
            list_R_value_y = []
            list_R_value = []
            for k in range(p_len):
                list_R_value_x.append(x_original_trajectories[IDs[i]][peds[j]][k+q+p_len] - x_predict_trajectories[IDs[i]][peds[j]][k])
                list_R_value_y.append(y_original_trajectories[IDs[i]][peds[j]][k+q+p_len] - y_predict_trajectories[IDs[i]][peds[j]][k])
                list_R_value.append(round(LA.norm([list_R_value_x[k], list_R_value_y[k]], 2), 4))
            R_value_x[IDs[i]].update({peds[j] : list_R_value_x})
            R_value_y[IDs[i]].update({peds[j] : list_R_value_y})
            R_value[IDs[i]].update({peds[j] : list_R_value})

    print("find_th_number_x length: " + str(len(find_th_number_x)))
    print("find_th_number_y length: " + str(len(find_th_number_y)))
    
    if not useMaxNorm:
        N = len(IDs) * len(peds) - len(find_th_number_x)
    else:
        N = len(IDs) - len(find_th_number_x)

    th_values = math.ceil((1+N)*(1-(0.05/H)))
    d_value_for_alpha_0_05 = []
    R = []
    numb_N = []
    numb_R = "R"

    for i in range(1,p_len+1):
        R.append(i)
        R[i-1] = numb_R + str(i)

    R_values = {}
    list_R_values = []

    for i in range(len(R)):
        R_values[R[i]] = []
        list_R_values = []

        if not useMaxNorm:
            for j in range(len(IDs)):
                for k in range(len(peds)):
                    list_R_values.append(R_value[IDs[j]][peds[k]][i])
        else:
            for j in range(len(IDs)):
                max_R = 0
                for k in range(len(peds)):
                    max_R = max(max_R, R_value[IDs[j]][peds[k]][i])
                list_R_values.append(max_R)
        list_R_values.sort()
        R_values.update({R[i]: list_R_values})
        d_value_for_alpha_0_05.append(R_values[R[i]][int(th_values)-1])
    

    d_value.update({d_list[q]: d_value_for_alpha_0_05})
    x_value.update({x_list[q]: x_original_trajectories})
    y_value.update({y_list[q]: y_original_trajectories})
    x_l_value.update({x_l_list[q]: x_predict_trajectories})
    y_l_value.update({y_l_list[q]: y_predict_trajectories})

    print(d_value_for_alpha_0_05)

    print(q, flush=True)

    break

end_time = time.time()

print("Total time: " + str(end_time-start_time))




import pickle

save_path = "/data2/mcleav/conformalRNNs/parameterizedScores/code/firstExample/dvalues/"
fid = open(save_path + "d_value.pkl", 'wb')
pickle.dump(d_value, fid)
fid.close()

fid = open(save_path + "x_value.pkl", 'wb')
pickle.dump(x_value, fid)
fid.close()

fid = open(save_path + "y_value.pkl", 'wb')
pickle.dump(y_value, fid)
fid.close()

fid = open(save_path + "x_l_value.pkl", 'wb')
pickle.dump(x_l_value, fid)
fid.close()

fid = open(save_path + "y_l_value.pkl", 'wb')
pickle.dump(y_l_value, fid)
fid.close()



