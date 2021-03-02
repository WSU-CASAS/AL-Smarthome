#!/usr/bin/python

# python al.py <option>+
#
# Performs activity learning on the given data file and outputs either the
# learned model, the results of a cross-validation test, or annotated data.

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import datetime
import gzip
import os
import sys
import warnings
from datetime import datetime
from datetime import timedelta

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

import config

cf = config.Config()
np.set_printoptions(threshold=sys.maxsize)
warnings.filterwarnings('ignore')


def get_datetime(date, time):
    """ Convert a pair of date and time strings to a datetime structure.
    """
    dtstr = date + ' ' + time
    try:
        dt = datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S.%f")
    except:
        dt = datetime.strptime(dtstr, "%Y-%m-%d %H:%M:%S")
    return dt


def compute_seconds(dt):
    """ Compute the number of seconds that have elapsed past midnight
    for the current datetime structure.
    """
    seconds = dt - dt.replace(hour=0, minute=0, second=0)
    return int(seconds.total_seconds())


def find_sensor(sensorname):
    """ Return the index of a specific sensor name in the list of sensors.
    """
    try:
        i = cf.sensornames.index(sensorname)
        return i
    except:
        print("Could not find sensor ", sensorname)
        return -1


def find_activity(aname):
    """ Return the index of a specific activity name in the list of activities.
    If the specified activity is not found in the list, add the new name to the
    list.
    """
    try:
        i = cf.activitynames.index(aname)
        return i
    except:
        if cf.mode == "TEST":
            print("Could not find activity ", aname)
            return -1
        else:
            cf.activitynames.append(aname)
            cf.num_activities += 1
            return cf.num_activities - 1


def read_data():
    """ Read the data file containing timestamped sensor readings, segment
    into non-overlapping windows, and extract features for each window.
    """
    first_event = True
    cf.current_timestamp = get_datetime("2001-01-01", "00:00:00.00000")

    datafile = open(cf.data_filename, "r")
    count = 0

    for line in datafile:
        words = line.split()  # Split line into words separated by spaces
        date = words[0]
        stime = words[1]
        sensorid = words[2]
        newsensorid = words[3]
        sensorstatus = words[4]
        alabel = words[5]
        dt = get_datetime(date, stime)
        cf.current_seconds_of_day = compute_seconds(dt)
        cf.day_of_week = dt.weekday()
        previousTimestamp = cf.current_timestamp
        cf.current_timestamp = dt

        snum1 = find_sensor(sensorid)
        timediff = cf.current_timestamp - previousTimestamp

        # reset the sensor times and the window
        if first_event or timediff.days < 0 or timediff.days > 1:
            for i in range(cf.num_sensors):
                cf.sensortimes[i] = cf.current_timestamp - timedelta(days=1)
            first_event = False
            cf.wincnt = 0

        if sensorstatus == "ON" and cf.dstype[snum1] == 'n':
            cf.dstype[snum1] = 'm'

        cf.sensortimes[snum1] = cf.current_timestamp  # last time sensor fired
        end = False
        tempdata = np.zeros(cf.num_features)
        if alabel != "Other_Activity" or not cf.ignore_other:
            end, tempdata = compute_feature(dt, sensorid, newsensorid)

        if end:  # End of window reached, add feature vector
            if alabel == "Other_Activity" and cf.cluster_other:
                cf.labels.append(-1)
            else:
                cf.labels.append(find_activity(alabel))
            cf.data.append(tempdata)
            count += 1

    datafile.close()
    if cf.mode != "TEST":
        if cf.cluster_other:
            cluster_other_class()


def cluster_other_class():
    """ Cluster the Other_Activity class into subclasses.
    """
    carray = []
    for i in range(len(cf.labels)):
        if cf.labels[i] == -1:  # Other activity
            carray.append(cf.data[i])
    kmeans = MiniBatchKMeans(n_clusters=cf.num_clusters).fit(carray)
    clabels = kmeans.labels_
    newlabels = kmeans.predict(carray)  # actual number of resulting clusters
    nc = len(set(newlabels))
    cf.num_clusters = nc
    for i in range(cf.num_clusters):
        cf.activitynames.append('cluster_' + str(i))

    index = 0  # assign Other_Activity labels new subclasses
    for i in range(len(cf.labels)):
        if cf.labels[i] == -1:  # Other activity
            cf.labels[i] = newlabels[index] + cf.num_activities
            index += 1
    with gzip.GzipFile(cf.model_path + 'clusters.gz', 'wb') as f:
        joblib.dump(kmeans, f)  # save clusters to file
    # joblib.dump(kmeans, cf.model_path + 'clusters.pkl')   # save clusters to file
    cf.num_activities += cf.num_clusters


def assign_features(wsize, prevwin1, prevwin2, sensorid1, lastlocation,
                    lastmotionlocation, complexity, numtransitions, numdistinctsensors):
    """ Assign values to features indices in the feature vector.
    """
    tempdata = np.zeros(cf.num_features)

    # Attribute 0..2: time of last sensor event in window
    tempdata[0] = cf.windata[cf.max_window - 1][1] / 3600  # hour of day
    tempdata[1] = cf.windata[cf.max_window - 1][1]  # seconds of day
    tempdata[2] = cf.windata[cf.max_window - 1][2]  # day of week

    # Attribute 3: time duration of window in seconds
    time1 = cf.windata[cf.max_window - 1][1]  # most recent sensor event
    time2 = cf.windata[cf.max_window - wsize][1]  # first sensor event in window
    if time1 < time2:
        duration = time1 + (cf.seconds_in_a_day - time2)
    else:
        duration = time1 - time2
    tempdata[3] = duration  # window duration

    timehalf = cf.windata[int(cf.max_window - (wsize / 2))][1]  # halfway point
    if time1 < time2:
        duration = time1 + (cf.seconds_in_a_day - time2)
    else:
        duration = time1 - time2
    if timehalf < time2:
        halfduration = timehalf + (cf.seconds_in_a_day - time2)
    else:
        halfduration = timehalf - time2
    if duration == 0.0:
        activitychange = 0.0
    else:
        activitychange = float(halfduration) / float(duration)

    # Attribute 4: time since last sensor event
    time2 = cf.windata[cf.max_window - 2][1]
    if time1 < time2:
        duration = time1 + (cf.seconds_in_a_day - time2)
    else:
        duration = time1 - time2
    tempdata[4] = duration

    # Attribute 5..6: dominant sensors from previous windows
    tempdata[5] = prevwin1
    tempdata[6] = prevwin2

    # Attribute 7: last sensor id in window
    tempdata[7] = find_sensor(sensorid1)

    # Attribute 8: last location in window
    tempdata[8] = lastlocation

    # Attribute 9: last motion location in window
    tempdata[9] = lastmotionlocation

    # Attribute 10: complexity (entropy of sensor counts)
    tempdata[10] = complexity

    # Attribute 11: activity change (activity change between window halves)
    tempdata[11] = activitychange

    # Attribute 12: number of transitions between areas in window
    tempdata[12] = numtransitions

    # Attribute 13: number of distinct sensors in window
    # tempdata[13] = numdistinctsensors
    tempdata[13] = 0
    return tempdata


def compute_feature(dt, sensorid1, sensorid2):
    """ Compute the feature vector for each window-size sequence of sensor events.
    The features (listed by index) are:
    0: time of the last sensor event in window (hour)
    1: time of the last sensor event in window (seconds)
    2: day of the week for the last sensor event in window
    3: window size in time duration
    4: time since last sensor event
    5: dominant sensor for previous window
    6: dominant sensor two windows back
    7: last sensor event in window
    8: last sensor location in window
    9: last motion sensor location in window
    10: complexity of window (entropy calculated from sensor counts)
    11: change in activity level between two halves of window
    12: number of transitions between areas in window
    13: number of distinct sensors in window
    14 - num_sensors+13: counts for each sensor
    num_sensors+14 - 2*num_sensors+13: time since sensor last fired (<= SECSINDAY)
    """
    lastlocation = -1
    lastmotionlocation = -1
    prevwin1 = prevwin2 = complexity = maxcount = 0
    numtransitions = numdistinctsensors = 0

    cf.windata[cf.wincnt][0] = find_sensor(sensorid1)
    cf.windata[cf.wincnt][1] = cf.current_seconds_of_day
    cf.windata[cf.wincnt][2] = cf.day_of_week

    if cf.wincnt < (cf.max_window - 1):  # not reached end of window
        cf.wincnt += 1
        return False, None
    else:  # reached end of window
        wsize = cf.max_window
        scount = np.zeros(cf.num_sensors, dtype=np.int)

        # Determine the dominant sensor for this window
        # count the number of transitions between areas in this window
        for i in range(cf.max_window - 1, cf.max_window - (wsize + 1), -1):
            scount[cf.windata[i][0]] += 1
            id = cf.windata[i][0]
            if lastlocation == -1:
                lastlocation = id
            if (lastmotionlocation == -1) and (cf.dstype[id] == 'm'):
                lastmotionlocation = id
            if i < cf.max_window - 1:  # check for transition
                id2 = cf.windata[i + 1][0]
                if id != id2:
                    if (cf.dstype[id] == 'm') and (cf.dstype[id2] == 'm'):
                        numtransitions += 1

        for i in range(cf.num_sensors):
            if scount[i] > 1:
                ent = float(scount[i]) / float(wsize)
                ent *= np.log2(ent)
                complexity -= float(ent)
                numdistinctsensors += 1

        if np.mod(cf.numwin, cf.max_window) == 0:
            prevwin2 = prevwin1
            prevwin1 = cf.dominant
            cf.dominant = 0
            for i in range(cf.num_sensors):
                if scount[i] > maxcount:
                    maxcount = scount[i]
                    cf.dominant = i

        tempdata = assign_features(wsize, prevwin1, prevwin2, sensorid1,
                                   lastlocation, lastmotionlocation, complexity, numtransitions,
                                   numdistinctsensors)
        # Attributes num_set_features..(num_sensors+(num_set_features-1))
        weight = 1
        for i in range(cf.max_window - 1, cf.max_window - (wsize + 1), -1):
            tempdata[cf.windata[i][0] + cf.num_set_features] += 1 * weight
            weight += cf.weightinc

        # Attributes num_sensors+num_set_features..
        #            (2*num_sensors+(num_set_features-1)) time since sensor fired
        for i in range(cf.num_sensors):
            difftime = cf.current_timestamp - cf.sensortimes[i]
            # There is a large gap in time or shift backward in time
            if difftime.total_seconds() < 0 or (difftime.days > 0):
                tempdata[cf.num_set_features + cf.num_sensors + i] = cf.seconds_in_a_day
            else:
                tempdata[cf.num_set_features + cf.num_sensors + i] = difftime.total_seconds()

        for i in range(cf.max_window - 1):
            cf.windata[i][0] = cf.windata[i + 1][0]
            cf.windata[i][1] = cf.windata[i + 1][1]
            cf.windata[i][2] = cf.windata[i + 1][2]
        cf.numwin += 1
        if cf.no_overlap and cf.mode != "ANNOTATE":
            cf.wincnt = 0
    return True, tempdata


def save_params():
    """ Save parameters to a file that will accompany a learned and saved model.
    """
    modelfilename = os.path.join(cf.model_path, cf.model_name + '.config')
    modelfile = open(modelfilename, "w")
    modelfile.write('python al.py --sensors \"[')
    for i in range(cf.num_sensors):  # sensor names
        modelfile.write(cf.sensornames[i])
        if i < (cf.num_sensors - 1):
            modelfile.write(",")
    modelfile.write(']\" ')
    modelfile.write('--activities \"[')
    for i in range(cf.num_activities):  # activity names
        modelfile.write(cf.activitynames[i])
        if i < (cf.num_activities - 1):
            modelfile.write(",")
    modelfile.write(']\" ')
    modelfile.write("--mode TEST ")
    if cf.cluster_other:
        modelfile.write("--clusterother ")
    elif cf.ignore_other:
        modelfile.write("ignoreother ")
    if cf.model_name != "model":
        modelfile.write('model ' + cf.model_name)
    modelfile.write("<datafilename>\n")
    modelfile.close()
    modelfilename = os.path.join(cf.model_path, cf.model_name + '.pkl.gz')
    with gzip.GzipFile(modelfilename, 'wb') as f:
        joblib.dump(cf.clf, f)  # save model to file


def report_results(xtest, ytest):
    """ Collect and report predictive accuracy of trained model.
    """
    numright = total = 0
    newlabels = cf.clf.predict(xtest)
    if cf.filter_other:  # do not count other in total
        matrix = np.zeros((cf.num_activities, cf.num_activities), dtype=int)
        for i in range(len(ytest)):
            if ytest[i] != -1:
                matrix[ytest[i]][newlabels[i]] += 1
                total += 1
                if newlabels[i] == ytest[i]:
                    numright += 1
    else:
        matrix = confusion_matrix(ytest, newlabels, labels=cf.activitynames)
        for i in range(len(ytest)):
            if not cf.cluster_other:
                if newlabels[i] == ytest[i]:
                    numright += 1
            else:
                if newlabels[i] > (cf.num_activities - (cf.num_clusters + 1)):
                    if ytest[i] == -1 or \
                            ytest[i] > (cf.num_activities - (cf.num_clusters + 1)):
                        numright += 1
                elif newlabels[i] == ytest[i]:
                    numright += 1
            total = len(ytest)
    print('activities', cf.activitynames)
    print('matrix\n', matrix)
    print('numright', numright, 'total', total)
    print(classification_report(ytest, newlabels))
    accuracy = float(numright) / float(total)
    return accuracy


def leave_one_out(files):
    """ Perform leave-one-subject-out testing. Assume each subject is represented
    by a specified file.
    """
    results = []
    for datafilename in files:
        print(datafilename)
        cf.data_filename = datafilename
        read_data()
        xtest = cf.data
        ytest = cf.labels
        k = len(cf.data[0])
        xtrain = np.empty((0, k), dtype=float)
        ytrain = np.empty((0), dtype=int)
        for otherfilename in files:
            if otherfilename != datafilename:
                cf.data = []
                cf.labels = []
                cf.data_filename = otherfilename
                read_data()
                xtrain = np.append(xtrain, cf.data, axis=0)
                ytrain = np.append(ytrain, cf.labels)
        cf.clf.fit(xtrain, ytrain)
        results.append(report_results(xtest, ytest))
    print('results', results)


def train_model():
    results = []
    if cf.add_pca:
        pca = PCA(n_components=50)
        pca_data = pca.fit_transform(cf.data)
        cf.data = np.append(cf.data, pca_data, axis=1)
    if cf.mode == "TRAIN":
        cf.clf.fit(cf.data, cf.labels)
    elif cf.mode == "CV":
        for i in range(3):
            xtrain, xtest, ytrain, ytest = train_test_split(cf.data,
                                                            cf.labels,
                                                            test_size=0.33,
                                                            random_state=i)
            cf.clf.fit(xtrain, ytrain)
            results.append(report_results(xtest, ytest))
        print('results', results)
    elif cf.mode == "PARTITION":
        dlength = len(cf.data)
        splitpoint = int((2 * dlength) / 3)
        xtrain = cf.data[:splitpoint]
        ytrain = cf.labels[:splitpoint]
        xtest = cf.data[splitpoint:]
        ytest = cf.labels[splitpoint:]
        cf.clf.fit(xtrain, ytrain)
        print('results ', report_results(xtest, ytest))
    elif cf.mode == "WRITE":
        labels = np.reshape(cf.labels, (len(cf.labels), 1))
        cf.data = np.append(cf.data, labels, 1)
        outputfilename = "./data.csv"
        np.savetxt(outputfilename, cf.data)


def test_model():
    read_data()
    modelfilename = os.path.join(cf.model_path, cf.model_name + '.pkl.gz')
    with open(modelfilename, 'rb') as f:
        cf.clf = joblib.load(f)
    print('accuracy ', report_results(cf.data, cf.labels))


def annotate_data(filename):
    """ Add activity labels to an input file containing sensor readings.
    """
    datafile = open(filename, "r")
    date = ""
    stime = ""
    modelfilename = os.path.join(cf.model_path, cf.model_name + '.pkl.gz')
    with open(modelfilename, 'rb') as f:
        cf.clf = joblib.load(f)
    outputfilename = "./data.al"
    outputfile = open(outputfilename, "w")
    cf.current_timestamp = get_datetime("2001-01-01", "00:00:00.00000")
    first_event = True
    fulldata = []
    for line in datafile:
        words = line.split()  # Split line into words on delimiter " "
        date = words[0]
        stime = words[1]
        sensorid = words[2]
        newsensorid = words[3]
        sensorstatus = words[4]
        dt = get_datetime(date, stime)
        cf.current_seconds_of_day = compute_seconds(dt)
        cf.day_of_week = dt.weekday()
        previousTimestamp = cf.current_timestamp
        cf.current_timestamp = dt

        snum1 = find_sensor(sensorid)

        timediff = cf.current_timestamp - previousTimestamp
        if first_event == True or timediff.days < 0 or timediff.days > 1:
            for i in range(cf.num_sensors):
                cf.sensortimes[i] = cf.current_timestamp - timedelta(days=1)
            first_event = False

        if sensorstatus == "ON" and cf.dstype[snum1] == 'n':
            cf.dstype[snum1] = 'm'

        cf.sensortimes[snum1] = cf.current_timestamp  # last time sensor fired

        end, tempdata = compute_feature(dt, sensorid, newsensorid)

        if end:  # End of window reached, add feature vector
            fulldata.append(tempdata)

    predict_alabel = cf.clf.predict(fulldata)
    datafile.close()

    datafile = open(filename, "r")
    linenum = 0
    for line in datafile:
        words = line.split()  # Split line into words on delimiter " "
        date = words[0]
        stime = words[1]
        sensorid = words[2]
        newsensorid = words[3]
        sensorstatus = words[4]
        if linenum < cf.max_window:
            aname = "Other_Activity"
        else:
            aname = cf.activitynames[predict_alabel[linenum - cf.max_window]]
            if aname.startswith("cluster_"):
                aname = "Other_Activity"
        outstr = date + " " + stime + " " + sensorid + " " + newsensorid + " "
        outstr += sensorstatus + " " + aname + "\n"
        outputfile.write(outstr)
        linenum += 1
    outputfile.close()


def main(args):
    files = cf.set_parameters()
    cf.num_features = cf.num_set_features + (2 * cf.num_sensors)
    if cf.mode == "TEST":
        test_model()
    elif cf.mode == "ANNOTATE":
        annotate_data(cf.data_filename)
    else:  # TRAIN, CV, WRITE
        if not os.path.exists(cf.model_path):  # create directory to store model file
            os.makedirs(cf.model_path)
        if cf.mode == "LOO":
            leave_one_out(files)
        else:
            read_data()
            print("finished reading data")
            train_model()
            if cf.mode == "TRAIN":
                save_params()


if __name__ == "__main__":
    main(sys.argv)
