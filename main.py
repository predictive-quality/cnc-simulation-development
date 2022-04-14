# Copyright (c) 2022 RWTH Aachen - Werkzeugmaschinenlabor (WZL)
# Contact: Simon Cramer, s.cramer@wzl-mq.rwth-aachen.de

import uuid
from datetime import datetime

import numpy as np
from absl import app
from absl import flags
from absl import logging

from simu import *

FLAGS = flags.FLAGS         # define parameters
flags.DEFINE_integer('runs', None, 'number of simulations')
flags.DEFINE_integer('x_val', '50', 'length of rectangle in mm')
flags.DEFINE_integer('y_val', '20', 'height of rectangle in mm')
flags.DEFINE_float('v_x', '0.15', 'moving speed x-achis in mm/s')
flags.DEFINE_float('v_y', '0.15', 'moving speed y-achis in mm/s')
flags.DEFINE_integer('sp_vol', '100', 'initial spindle voltage')
flags.DEFINE_integer('fd_vol', '200', 'initial feed rate voltage')
flags.DEFINE_float('restn', '0.125', 'time resolution in seconds')
flags.DEFINE_float('err_magnitude', '0.02', 'numerical factor to  set magnitude of normal error')
flags.DEFINE_float('err_initial', '0.125', 'initial error magnitude of feed_rate and rotation_speed')
flags.DEFINE_integer('err_typ', '0', 'type of error')       # 0 for normal deviation, 1 for spikes
flags.DEFINE_boolean('plot_runs', 'False', 'plot runs')


# Required flag.
flags.mark_flag_as_required("runs")     # run by adding --runs= number of runs in terminal


def main(argv):
    del argv

    logging.info("simulation started...")
    data = []
    time = datetime.now()
    for i in range(FLAGS.runs):
        feed_rate = np.maximum(0.2, random.gauss(1, FLAGS.err_initial))  # initial feed_rate
        rotation_speed = feed_rate# np.maximum(0.2, random.gauss(1, FLAGS.err_initial))  # initial rotation_speed

        # simulation run
        rectangle = create_rectangle(FLAGS.err_magnitude, str(uuid.uuid4())[-6:], FLAGS.x_val, FLAGS.y_val, FLAGS.v_x, FLAGS.v_y, FLAGS.sp_vol,
                             FLAGS.fd_vol, FLAGS.restn, rotation_speed, feed_rate, FLAGS.err_typ, time)
        data.append(rectangle)

        time_s = np.round((FLAGS.x_val / FLAGS.v_x + FLAGS.y_val / FLAGS.v_y) * 2) + 60  # next time point + 60s time
        time = time + timedelta(seconds=time_s)

    input_data = {'id': [], 'time': [], 'kind': [], 'value': []}
    output_data = {'id': [], 'time': [], 'kind': [], 'value': []}

    for d in data:
        if FLAGS.plot_runs:
            plot_path(d)

        # append data to dict
        for i in ['x_ist', 'y_ist', 'x_soll', 'y_soll', 'spindel_voltage', 'feed_voltage', 'rot_speed', 'sp_fwd']:
            input_data['id'].extend([d['id'] for k in d['time']])
            input_data['time'].extend(d['time'])
            input_data['kind'].extend([i for k in d['time']])
            input_data['value'].extend(d[i])

        # 3 qualitycharacteristics
        output_data['id'].append(d['id'])
        output_data['value'].append(d['meas1'])
        output_data['time'].append(d['meas_time'])
        output_data['kind'].append('QC1')

        output_data['id'].append(d['id'])
        output_data['value'].append(d['meas2'])
        output_data['time'].append(d['meas_time'])
        output_data['kind'].append('QC2')

        output_data['id'].append(d['id'])
        output_data['value'].append(d['meas3'])
        output_data['time'].append(d['meas_time'])
        output_data['kind'].append('QC3')

    logging.info("simulation done. saving...")

    input_data = pd.DataFrame.from_dict(input_data)
    output_data = pd.DataFrame.from_dict(output_data)

    input_data['time'] = input_data["time"].dt.strftime('%Y-%m-%d %H:%M:%S.%f').apply(
        lambda x: x[0:10] + "T" + x[11:] + "Z")
    output_data['time'] = output_data["time"].dt.strftime('%Y-%m-%d %H:%M:%S.%f').apply(
        lambda x: x[0:10] + "T" + x[11:] + "Z")

    input_data.to_csv('input.csv', sep=',', index=False)
    output_data.to_csv('output.csv', sep=',', index=False)

    logging.info("saved.")


if __name__ == "__main__":
    app.run(main)
