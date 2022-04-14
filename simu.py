# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

import random
from datetime import timedelta
from absl import logging

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.random import normal
from plotly.subplots import make_subplots


def move(delta_x, delta_y, path, v_x_max, v_y_max, sp_vol, fd_vol, rotation_speed, feed_rate,
         delta_t, err, id, err_typ, spindle_multipicator=134.6, feed_multiplicator=143.7):
    """ simulation with the given parameters

    using the defined parameters to simulate the milling process for equidistant timesteps
    determined be the speed and distance. Moving in one direction.

    Args:
        delta_x: integer distance in x direction
        delta_y: integer distance in y direction
        v_x_max: float maximal x speed
        v_y_max: float maximal y speed
        sp_vol: integer spindle voltage
        fd_vol: integer feed voltage
        rotation_speed: float initial rotation_speed
        feed_rate: float initial feed_rate
        delta_t: float time resolution in seconds
        err: float determines the deviation from theoretical values
        id: id of the run
        err_typ: int type of error, either constant error or spikes
        spindle_multipicator: spindle_multipicator, default = 134.6
        feed_multiplicator: feed_multiplicator = 143.7
    Returns:
        path: a dict with all the calculated values for all timepoints
    """

    t_x = abs(delta_x) / v_x_max  # time needed for x
    t_y = abs(delta_y) / v_y_max  # time needed for y

    # recalculating speeds, based on maximum velocity
    t = np.maximum(t_x, t_y)
    v_x_actual = delta_x / t
    v_y_actual = delta_y / t

    # initialize time array and distances
    t_arranged = np.arange(0, t+delta_t, delta_t)
    distance_x = v_x_actual * t_arranged
    distance_y = v_y_actual * t_arranged

    # calculate setpoints from x, y starting values
    x_soll = distance_x + path['x_soll'][-1]
    y_soll = distance_y + path['y_soll'][-1]

    # define timepoints
    time = path['time'][-1] + np.arange(len(t_arranged)) * timedelta(seconds=delta_t)

    # calculation of spindle voltage and feed voltage
    error_scale_spindel_voltage = 0.2
    spindel_voltage = path['spindel_voltage'][-1] + np.sqrt(delta_t) * err * error_scale_spindel_voltage * np.cumsum(
        np.random.normal(0, 1, size=distance_x.size) * spindle_multipicator)
    feed_voltage = path['feed_voltage'][-1] + np.sqrt(delta_t) * err * error_scale_spindel_voltage * np.cumsum(
        np.random.normal(0, 1, size=distance_x.size) * feed_multiplicator)

    # calculation of the deviation from the optimal trajectory determined by error type
    #if err_typ == 0:   # normal deviation each timestep
        # rotation speed and feed_rate
    error_scale_rot_sp = 0.01
    rot_speed = path['rot_speed'][-1] + np.sqrt(delta_t) * err * error_scale_rot_sp * np.cumsum(
        np.random.normal(0, 1, size=distance_x.size))
    sp_fwd = path['sp_fwd'][-1] + np.sqrt(delta_t) * err * error_scale_rot_sp * np.cumsum(
        np.random.normal(0, 1, size=distance_x.size))

    if err_typ == 1:   # error Spike randomly in rotation speed and/or feed_rate
        if np.random.rand() < .15:
            start_err = random.randint(0, np.floor(distance_x.size*0.5))
            end_err = np.minimum(distance_x.size-1,start_err+np.random.randint(np.floor(distance_x.size*0.10), np.ceil(distance_x.size*0.25)))
            err_val = (end_err-start_err)/distance_x.size*np.sign(np.random.normal(0, 1))
            err_spike = np.ones(distance_x.size)
            err_spike[start_err:end_err] = 1+err_val
            if np.random.rand() < 0.5:
                rot_speed = np.maximum(0.2, err_spike * rot_speed)
            else:
                sp_fwd = np.maximum(0.2, err_spike * sp_fwd)


    # absolute error depending on the ratio of rot_speed and sp_fwd for vector of values
    get_error_scale_numpy = np.vectorize(get_error_scale)
    err_sc = get_error_scale_numpy(rot_speed, sp_fwd)

    # error direction
    get_direction_numpy = np.vectorize(get_error_direction)
    direction = get_direction_numpy(rot_speed, sp_fwd)

    # error based on the ratio
    x_err = np.cumsum((np.ones(distance_x.size) - err_sc) * direction * v_x_actual) * err * 0.2
    y_err = np.cumsum((np.ones(distance_y.size) - err_sc) * direction * v_y_actual) * err * 0.2

    # normal and random error each timestep
    # changed sqrt(delta_t) to delta_t: scaling with the timeframe. Smaller timeframe leads otherwise to smaller
    # error and vise versa since the cumsum formula adds error per timestep.
    multiplikator_err_nor = 0.005
    x_err_nor = np.cumsum(np.random.normal(np.zeros(distance_x.size), 1)) * multiplikator_err_nor * err
    y_err_nor = np.cumsum(np.random.normal(np.zeros(distance_y.size), 1)) * multiplikator_err_nor * err
    multiplikator_err_rand = 0
    x_err_rand = np.random.normal(np.zeros(distance_x.size), 1) * multiplikator_err_rand * err
    y_err_rand = np.random.normal(np.zeros(distance_y.size), 1) * multiplikator_err_rand * err

    # real x and y values including errors
    x_ist = distance_x + path['x_ist'][-1] + np.sqrt(delta_t) * (- x_err + x_err_nor + x_err_rand)
    y_ist = distance_y + path['y_ist'][-1] + np.sqrt(delta_t) * (- y_err + y_err_nor + y_err_rand)

    # Append values to path variable
    path['x_soll'].extend(x_soll.tolist())
    path['y_soll'].extend(y_soll.tolist())
    path['x_ist'].extend(x_ist.tolist())
    path['y_ist'].extend(y_ist.tolist())
    path['time'].extend(time.tolist())
    path['spindel_voltage'].extend(spindel_voltage.tolist())
    path['feed_voltage'].extend(feed_voltage.tolist())
    path['rot_speed'].extend(rot_speed.tolist())
    path['sp_fwd'].extend(sp_fwd.tolist())
    path['ratio'].extend(err_sc.tolist())

    return path


def create_rectangle(err, id, x_val, y_val, v_x, v_y, sp_vol, fd_vol, restn, rotation_speed, feed_rate, const_err, time):
    """ milling of a rectangle

    milling of a rectangle by moving along the 4 lines of a rectangle. Move function is applied 4 times and
    testpoints for the quality measurements are defined.

    Args:
        err: scaling factor for err intensity
        id: id of the run
        x_val: integer length of the x side of the rectangle
        y_val: integer length of the y side of the rectangle
        v_x: float maximal speed in x direction
        v_y: float maximal speed in y direction
        sp_vol: integer spindle voltage
        fd_vol: integer feed voltage
        restn: float time resolution in seconds
        rotation_speed: float rotation speed
        feed_rate: float feed rate
        const_err: in err type
        time: time value of the start of the simulation

    Returns: dict of all points and values in the rectangle run

    """

    # testpoints for quality measurements; 3 points are defined
    testpoint1 = x_val / 2.25
    testpoint2_x = x_val
    testpoint2_y = y_val
    testpoint3 = y_val / 2.25

    # initial conditions
    path = {'id': id, 'time': [time], 'x_ist': [0.0], 'y_ist': [0.0], 'x_soll': [0.0], 'y_soll': [0.0],
            'spindel_voltage': [sp_vol], 'feed_voltage': [fd_vol], 'rot_speed': [rotation_speed], 'sp_fwd': [feed_rate],
            'ratio': [get_error_scale(rotation_speed, feed_rate)]}

    # make square
    path = move(x_val, 0, path, v_x, v_y, sp_vol, fd_vol, rotation_speed, feed_rate, restn, err, id, const_err)
    path = move(0, y_val, path, v_x, v_y, sp_vol, fd_vol, rotation_speed, feed_rate, restn, err, id, const_err)
    path = move(-x_val, 0, path, v_x, v_y, sp_vol, fd_vol, rotation_speed, feed_rate, restn, err, id, const_err)
    path = move(0, -y_val, path, v_x, v_y, sp_vol, fd_vol, rotation_speed, feed_rate, restn, err, id, const_err)

    # calculate quality measurements and append to dict
    path['meas1'] = get_measurement(path, testpoint1, None)
    path['meas2'] = get_measurement(path, testpoint2_x, testpoint2_y)
    path['meas3'] = get_measurement(path, None, testpoint3)
    path['meas_time'] = time

    return path


def plot_path(path):
    """ make plotly plots of process parameters

    Args:
        path: dict with values of variables and parameters for all timepoints

    """
    df = pd.DataFrame.from_dict(path)
    fig = make_subplots(rows=5, cols=1)

    fig.add_trace(go.Scatter(y=df['x_ist'], x=df['time'],
                             mode='lines',
                             name='x_ist'), row=1, col=1)
    fig.add_trace(go.Scatter(y=df['x_soll'], x=df['time'],
                             mode='lines',
                             name='x_soll'), row=1, col=1)
    fig.add_trace(go.Scatter(y=df['y_ist'], x=df['time'],
                             mode='lines',
                             name='y_ist'), row=1, col=1)
    fig.add_trace(go.Scatter(y=df['y_soll'], x=df['time'],
                             mode='lines',
                             name='y_soll'), row=1, col=1)

    fig.add_trace(go.Scatter(y=df['x_ist']-df['x_soll'], x=df['time'],
                             mode='lines',
                             name='x_delta'), row=2, col=1)
    fig.add_trace(go.Scatter(y=df['y_ist']-df['y_soll'], x=df['time'],
                             mode='lines',
                             name='y_delta'), row=2, col=1)

    fig.add_trace(go.Scatter(y=df['spindel_voltage'], x=df['time'],
                             mode='lines',
                             name='spindel_voltage'), row=3, col=1)
    fig.add_trace(go.Scatter(y=df['feed_voltage'], x=df['time'],
                             mode='lines',
                             name='feed_voltage'), row=3, col=1)
    fig.add_trace(go.Scatter(y=df['rot_speed'], x=df['time'],
                             mode='lines',
                             name='rot_speed'), row=4, col=1)
    fig.add_trace(go.Scatter(y=df['sp_fwd'], x=df['time'],
                             mode='lines',
                             name='sp_fwd'), row=4, col=1)
    fig.add_trace(go.Scatter(y=df['ratio'], x=df['time'],
                             mode='lines',
                             name='ratio'), row=5, col=1)

    fig.show()


def get_measurement(path, testpoint_x, testpoint_y):
    """ calculate quality measurement

    calculate the distance between to opposite points in the rectangle.

    Args:
        path: dict with values of variables and parameters for all timepoints
        testpoint_x: float point where to measure x value
        testpoint_y: float point where to measure y value

    Returns:
        distance: float distance between points on opposite sides of rectangle
    """

    # finding point in array for measurement point in x direction
    index = 1
    if testpoint_x is None:
        test_indexes = np.abs(np.array(path['y_soll']) - testpoint_y) < 0.1  # 0.01
        test_indexes = np.where(test_indexes)
    elif testpoint_y is None:
        test_indexes = np.abs(np.array(path['x_soll']) - testpoint_x) < 0.1  # 0.01
        test_indexes = np.where(test_indexes)
    else:
        var1 = np.all([np.abs(np.array(path['y_soll']) - 0) < 0.1,
                       np.abs(np.array(path['x_soll']) - testpoint_x) < 0.1], axis=0)
        var2 = np.all([np.abs(np.array(path['y_soll']) - testpoint_y) < 0.1,
                       np.abs(np.array(path['x_soll']) - 0) < 0.1], axis=0)

        test_indexes = np.any([var1, var2], axis=0)
        test_indexes = np.where(test_indexes)
        index = 2

    x_1 = path['x_ist'][test_indexes[0][0]]
    x_2 = path['x_ist'][test_indexes[0][index]]

    y_1 = path['y_ist'][test_indexes[0][0]]
    y_2 = path['y_ist'][test_indexes[0][index]]

    # calculate distance
    meas = np.sqrt(np.power(x_1-x_2, 2)+np.power(y_1-y_2, 2))
    distance = np.round(meas, 8)
    return distance


def get_error_direction(rotation_speed: float, feed_rate: float) -> float:
    """ calculation of the error deviation

    calculation of the error deviation based on the ratio of rotation_speed and feed_rate

    Args:
        rotation_speed: rotation_speed float
        feed_rate: feed_rate float

    Returns:
         value of 1 or -1 indicating the direction of the deviation
    """
    ratio = rotation_speed / feed_rate

    if ratio > 1:
        return -1
    return 1


def get_error_scale(rotation_speed: float, feed_rate: float) -> float:
    """ calculation of the absolut error

    calculation of the error size based on the ratio of rotation_speed and feed_rate

    Args:
        rotation_speed: rotation_speed float
        feed_rate: feed_rate float

    Returns:
         float value depending on the ratio

    Raises:
         AttributeError: if ratio is off
    """
    ratio = rotation_speed / feed_rate

    return ratio
