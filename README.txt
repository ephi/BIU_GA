#################### Usage Instructions #######################
environment: python 3.8 (3.9 is prefered), numpy, graycode, matplotlib
This project includes 3 scripts:
GAFrameWork.py - which is the GA framework script, imported for the different problems
GANQueens.py   - which contains the GA solution for N(=8)-Queen problem
GATSP.py       - which contains the GA solution for the TS problem
This probject also includes:
PythonGALib.pyd - which is a CPP library that allows quick graycode functions (used with python 3.9 instead of graycode package).
all of the files are exepcted to be located at the same folder

Also, it is expected that tsp.txt which describes the TSP will be located at the same folder as GATSP.py
##################### GANQueens.py ############################
Usage: python GANQueen.py
Output: GA solution (chromosme print, GA graphs), CS-Randomized solution (solution print)
##################### GATSP.py ################################
Usage: python GATSP.py
Output: GA solution (a list of cities into tsp_output.txt, GA graphs if SHOW_GRAPH = True, best fitness and time measure print).
Notes: the GA algorithm will run for 
       min ( # of generations by NUM_OF_GENERATIONS, 
             # of generations until BEST_FITNESS_THRESHOLD is met, 
             # of generations until RUNTIME_THRESHOLD_IN_SEC is met
           )




