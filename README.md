# CNC Simulation

This simulation generates datapoints for a simultion of an rectangular milling process.


## Installation

Clone the repository and install all requirements using
> pip install -r requirements.txt

## Usage

You can run the code in two ways.
- Use command line flags as arguments:
> python main.py --runs=10 --plot_runs=...
- Use a flagfile.txt which includes the arguments:
 > python main.py --flagfile=example/flagfile.txt

## Input Flags/Arguments:
#### --runs
 number of runs of simulation, integer, required
#### --x_val
length of rectangle in mm, interger, default = 50
#### --y_val
width of rectangle in mm, interger, default = 20
#### --v_x
moving speed x-achis in mm/s, float, default = 0.15
#### --v_y
moving speed y-achis in mm/s, float, default = 0.15
#### --sp_vol
initial spindle voltage, integer, default = 100
#### --fd_vol
initial feed rate voltage, integer, default = 200
#### --restn
time resolution in seconds, float, default = 0.125
#### --err_magnitude
value for the magnitude of the error, increase for larger errors, default = 0.02
#### --err_initial
parameter for the deviation of the initial values of feed_rate and rotation_speed, default = 0.125
#### --err_typ
type of error, 0 or 1, default 0

Type of errors are purely random. Type 1 errors induce spikes in the Spindel Voltage and Feed Rate leading to deviations in x/y.

#### --plot_runs
Boolean if runs should be plotted, default = False

## Example
First move to the repository directory.
We run e.g. the simulation for 10 runs by running:
> python main.py --flagfile=example/flags.txt

For plotting the runs and for a larger rectangle, respectively, we can use the following commands:
> python main.py --flagfile=example/flagfile_plot.txt

> python main.py --flagfile=example/flagfile_larger.txt

Another example where the error has a spike form can be used by running:
> python main.py --flagfile=example/flagfile_spikes.txt
