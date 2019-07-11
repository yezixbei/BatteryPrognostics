
import glob, string, collections, math

import tensorflow as tf
import numpy as np

from scipy.io import loadmat, whosmat
import matplotlib.pyplot as plt
import datetime
import json


FILE_NAME = 'B0007_charge.json'
BATTERY_NAME = 'B0007'
KEEP_RATIO = 0.8
TRAIN_RATIO = 0.7
STEP_SZ = 10 # number of windows needed to predict one window ahead
WINDOW_SZ = 1 # sliding window size
BATCH_SZ = 1 
LEARN_RATE = 1e-3
RNN_SIZE = 128




# DATA CLEANING #

# helpers
def removeOutliers(x, y, segSz=10):
    x_res = []
    y_res = []
    n = len(y)/segSz
    for i in xrange(0, n*segSz, segSz):
        yy = np.array(y[i:i+segSz])
        xx = np.array(x[i:i+segSz])
        idx = np.abs(yy-np.mean(yy)) < 2*np.std(yy)
        x_res.extend(xx[idx].tolist())
        y_res.extend(yy[idx].tolist())
    return x_res, y_res

def extractCapacity(data):
    cap = [] 
    cyc = []
    for key in sorted(data.iterkeys()):
        cyc.append(int(key))
        cap.append(np.trapz(data[key]["current_battery"], data[key]["time"]))
        plt.plot(data[key]["time"], data[key]["current_battery"], '-')
        plt.ylabel('Current (A)')
        plt.xlabel('Time (s)')
    
    plt.axis([0, 11000, 0, 1.6])
    plt.title('Charge Current v. Time - ' + BATTERY_NAME)
    plt.show()

    return cap, cyc

# CTR is Charge Transfer Resistance
def extractCTR(data): 
    ctr = [] 
    cyc = []
    for key in sorted(data.iterkeys()):
        cyc.append(int(key))
        ctr.append(data[key]["rct"])

    return ctr, cyc

def parse(filename):
    with open(filename) as f:    
        data = json.load(f)

    fileType = filename.split("_")[1].split(".")[0]

    if fileType == 'impedance': 
        y, x = extractCTR(data)
    else:
        y, x = extractCapacity(data)

    yMax = np.amax(y)
    keepIndx = int(KEEP_RATIO *len(y))
    y = (np.array(y)/yMax).tolist()[:keepIndx]
    x = x[:keepIndx]

    # Plot Input Before Smoothing
    plt.plot(x, y, 'o', markersize=4)
    plt.axis([0, 550, 0.5, 1])
    plt.ylabel('Normalized Capacity (1/Ah)')
    plt.xlabel('Cycle')
    plt.title('Before Smoothing - ' + BATTERY_NAME)
    plt.show()

    x, y = removeOutliers(x, y)

    # Fig. 1 -- Plot Input After Smoothing
    plt.plot(x, y, 'o', markersize=4)
    plt.axis([0, 550, 0.5, 1])
    plt.ylabel('Normalized Capacity (1/Ah)')
    plt.xlabel('Cycle')
    plt.title('Charge Capacity v. Cycle - ' + BATTERY_NAME)
    plt.show()


    splitIndx = int(TRAIN_RATIO *len(y))
    trainX = x[:splitIndx]
    testX = x[splitIndx:]
    trainY = y[:splitIndx]
    testY = y[splitIndx:]
    
    return trainX, trainY, testX, testY

def batch(series):
    # Split into n non-overlapping windows
    n = len(series)/WINDOW_SZ
    win = []
    for i in xrange(0, n*WINDOW_SZ, WINDOW_SZ):
        win.append(series[i:i+WINDOW_SZ])

    # Split windows into data and labels
    inputs = []
    labels = []
    for i in xrange(0, len(win)-STEP_SZ, 1):
        inputs.append(win[i:i+STEP_SZ])
        labels.append(win[i+STEP_SZ])
    
    # Batch
    b = len(inputs)/BATCH_SZ
    bInputs = []
    bLabels = []
    for i in xrange(0, b*BATCH_SZ, BATCH_SZ):
        bInputs.append(inputs[i:i+BATCH_SZ])
        bLabels.append(labels[i:i+BATCH_SZ])
    return bInputs, bLabels




# TENSORFLOW #

# Inputs #
batch_in = tf.placeholder(tf.float32, shape=[BATCH_SZ, STEP_SZ, WINDOW_SZ])
labels = tf.placeholder(tf.float32, shape=[BATCH_SZ, WINDOW_SZ])

# RNN Layer #
rnn = tf.contrib.rnn.GRUCell(RNN_SIZE) 
outputs, final_state = tf.nn.dynamic_rnn(rnn, batch_in, initial_state=rnn.zero_state(BATCH_SZ, tf.float32)) 

# Linear Layer #
W = tf.Variable(tf.truncated_normal([RNN_SIZE, WINDOW_SZ], stddev=0.1))
b = tf.Variable(tf.truncated_normal([WINDOW_SZ], stddev=0.1))

logits = tf.add(tf.matmul(final_state, W), b)

# Backward Pass #
loss = tf.reduce_mean(tf.square(logits - labels))
trainOp = tf.train.RMSPropOptimizer(learning_rate=LEARN_RATE).minimize(loss)



if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Training #
    print "\n## Training ##"

    trainX, trainY, testX, testY = parse(FILE_NAME)
    trainD, trainL = batch(trainY)
    testD, testL = batch(testY)

    for step in xrange(len(trainD)):
        print "\nstep", step, "out of", len(trainD)-1

        feedict = {batch_in: trainD[step], labels: trainL[step]}
        logitsR, lossR, _ = sess.run([logits, loss, trainOp], feed_dict=feedict)

        print "loss -", lossR

    # Testing #
    print "\n## Testing ##"

    predicted = []
    actual = []
    for step in xrange(len(testD)):
        print "\nstep", step, "out of", len(testD)-1

        feedict = {batch_in: testD[step], labels: testL[step]}
        logitsR, lossR = sess.run([logits, loss], feed_dict=feedict)

        print "loss -", lossR

        predicted.append(np.ndarray.tolist(logitsR))
        actual.append(testL[step])

    predicted = np.array(predicted).flatten().tolist()
    actual = np.array(actual).flatten().tolist()

    # Plot Results
    plt.plot(trainX, trainY, 'bo', markersize=2)
    plt.plot(testX[-len(predicted):], actual, 'bx', label='Real Data', markersize=3)
    plt.plot(testX[-len(predicted):], predicted, 'rx', label='Predicted Values', markersize=3)
    plt.axhline(y=0.8, label='Failure Threshold')
    plt.legend()
    plt.title('Charge Capacity v. Cycle - ' + BATTERY_NAME)
    plt.ylabel('Normalized Capacity (1/Ah)')
    plt.xlabel('Cycle')
    plt.axis([0, 550, 0.5, 1])
    plt.show()
    
    # Fig. 2 -- Results Zoomed In
    plt.plot(actual, 'bx', label='Real Data')
    plt.plot(predicted, 'rx', label='Predicted Values')
    plt.axhline(y=0.8, label='Failure Threshold')
    plt.legend()
    plt.title('Charge Capacity v. Cycle - ' + BATTERY_NAME)
    plt.ylabel('Normalized Capacity (1/Ah)')
    plt.xlabel('Cycle (From Beginning of Test Data)')
    plt.show()

