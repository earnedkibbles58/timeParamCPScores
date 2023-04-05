# timeParamCPScores

This is the code for the paper: "Conformal Prediction Regions for Time Series using Linear Complementarity Programming"



To use this repo, you will first need to setup gurobi (https://www.gurobi.com/) and install gurobipy

You will also need to install the following python packages: numpy, matplotlib, pickle, math, random, time

To generate the results from Section 6.2 go into the code directory and run

python runExpORCALSTMData.py


To generate the results from Section 6.3 go into the code directory and run

python checkCoveragesCPORCAData.py

Note that this script may take a few mins to run.
