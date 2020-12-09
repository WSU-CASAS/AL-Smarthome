# config.py
# Global variables

# Written by Diane J. Cook, Washington State University.

# Copyright (c) 2020. Washington State University (WSU). All rights reserved.
# Code and data may not be used or distributed without permission from WSU.


import argparse
import copy
import numpy as np
from sklearn.ensemble import RandomForestClassifier


MODE_TRAIN = 'TRAIN'
MODE_TEST = 'TEST'
MODE_CV = 'CV'
MODE_PARTITION = 'PARTITION'
MODE_ANNOTATE = 'ANNOTATE'
MODE_WRITE = 'WRITE'
MODE_LOO = 'LOO'
MODES = list([MODE_TRAIN,
              MODE_TEST,
              MODE_CV,
              MODE_PARTITION,
              MODE_ANNOTATE,
              MODE_WRITE,
              MODE_LOO])


class Config:

    def __init__(self):
        """ Constructor
        """
        self.activitynames = ['Bed_Toilet_Transition', 'Go_To_Sleep', 'Sleep', 'Wake_Up',
                              'Personal_Hygiene', 'Other_Activity', 'Bathe', 'Step_Out',
                              'Morning_Meds', 'Cook_Breakfast', 'Eat_Breakfast',
                              'Wash_Breakfast_Dishes', 'Dress', 'Toilet', 'Exercise', 'Watch_TV',
                              'Leave_Home', 'Enter_Home', 'Work_At_Table', 'Evening_Meds',
                              'Cook_Lunch', 'Eat_Lunch', 'Wash_Lunch_Dishes', 'Relax',
                              'Cook_Dinner', 'Eat_Dinner', 'Wash_Dinner_Dishes', 'Read',
                              'Sleep_Out_Of_Bed', 'Phone', 'Wash_Dishes', 'Cook', 'Eat']
        self.current_seconds_of_day = 0
        self.current_timestamp = 0
        self.day_of_week = 0
        self.dominant = 0
        self.sensornames = ['Bath', 'Bathroom', 'BathroomDoor', 'BathroomLight', 'Bed', 'Bedroom',
                            'BedroomDoor', 'BedroomLight', 'Chair', 'Closet', 'ClosetDoor',
                            'ClosetLight', 'Couch', 'Desk', 'DiningRoom', 'DiningRoomLight',
                            'Entry', 'FrontDoor', 'FrontDoorLight', 'GarageDoor', 'GuestBathroom',
                            'GuestRoom', 'Hall', 'HallLight', 'Kitchen', 'KitchenCabinet',
                            'KitchenLight', 'LaundryRoom', 'LivingRoom', 'LivingRoomLight',
                            'LoungeChair', 'MedCabinet', 'Office', 'OfficeLight', 'OtherRoom',
                            'OutsideDoor', 'Refrigerator', 'Sink', 'Stove', 'TV', 'Toilet',
                            'WorkArea', 'WorkAreaLight']
        self.sensortimes = []
        self.data = []
        self.dstype = []
        self.labels = []
        self.numwin = 0
        self.wincnt = 0
        self.data_filename = "data"
        self.filter_other = True  # Do not consider other in performance
        self.ignore_other = False
        self.cluster_other = False
        self.no_overlap = False  # Do not allow windows to overlap
        self.confusion_matrix = True
        self.mode = MODE_TRAIN  # TRAIN, TEST, CV, PARTITION, ANNOTATE, WRITE, LOO
        self.model_path = "./model/"
        self.model_name = "model"
        self.num_activities = 0
        self.num_clusters = 10
        self.num_sensors = 0
        self.num_set_features = 14
        self.features = 0
        self.num_features = 0
        self.seconds_in_a_day = 86400
        self.max_window = 30
        self.add_pca = False  # Add principal components to feature vector
        self.weightinc = 0.01
        self.windata = np.zeros((self.max_window, 3), dtype=np.int)
        self.clf = RandomForestClassifier(n_estimators=80,
                                          max_features=8,
                                          bootstrap=True,
                                          criterion="entropy",
                                          min_samples_split=20,
                                          max_depth=None,
                                          n_jobs=4,
                                          class_weight='balanced')

    def set_parameters(self):
        """ Set parameters according to command-line args list.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode',
                            dest='mode',
                            type=str,
                            choices=MODES,
                            default=self.mode,
                            help=('Define the core mode that we will run in, default={}.'
                                  .format(self.mode)))
        parser.add_argument('--data',
                            dest='data',
                            type=str,
                            default=self.data_filename,
                            help=('Data file of sensor data, default={}'
                                  .format(self.data_filename)))
        parser.add_argument('--model',
                            dest='model',
                            type=str,
                            default=self.model_name,
                            help=('Specifies the name of the model to use, default={}'
                                  .format(self.model_name)))
        parser.add_argument('--ignoreother',
                            dest='ignoreother',
                            default=self.ignore_other,
                            action='store_true',
                            help=('Ignores all sensor events affiliated with activity '
                                  'Other_Activity, default={}'.format(self.ignore_other)))
        parser.add_argument('--clusterother',
                            dest='clusterother',
                            default=self.cluster_other,
                            action='store_true',
                            help=('Divides the Other_Activity category into subclasses using '
                                  'k-means clustering.  When activated this sets --ignoreother '
                                  'to False, default={}'.format(self.cluster_other)))
        parser.add_argument('--sensors',
                            dest='sensors',
                            type=str,
                            default=','.join(self.sensornames),
                            help=('Comma separated list of sensors that appear in the data file, '
                                  'default={}'.format(','.join(self.sensornames))))
        parser.add_argument('--activities',
                            dest='activities',
                            type=str,
                            default=','.join(self.activitynames),
                            help=('Comma separated list of activities to use, '
                                  'default={}'.format(','.join(self.activitynames))))
        parser.add_argument('files',
                            metavar='FILE',
                            type=str,
                            nargs='*',
                            help='Data files for AL to process.')
        args = parser.parse_args()

        self.mode = args.mode
        self.data_filename = args.data
        self.model_name = args.model
        self.ignore_other = args.ignoreother
        self.cluster_other = args.clusterother
        if self.cluster_other:
            self.ignore_other = False
        self.sensornames = str(args.sensors).split(',')
        self.num_sensors = len(self.sensornames)
        for i in range(self.num_sensors):
            self.sensortimes.append(0)
            self.dstype.append('n')
        self.activitynames = str(args.activities).split(',')
        self.num_activities = len(self.activitynames)
        files = list()
        if len(args.files) > 0:
            files = copy.deepcopy(args.files)
        else:
            files.append(self.data_filename)

        return files
