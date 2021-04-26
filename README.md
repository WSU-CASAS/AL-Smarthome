# AL Activity Learning - Smart Home

The AL activity learner (smart home edition) learns activity models from
activity-labeled ambient sensor data. The learned models can be used to label
new ambient sensor data with corresponding activity labels.

Author: Dr. Diane J. Cook, School of Electrical Engineering and
Computer Science, Washington State University, email: djcook@wsu.edu.

Support: This material is based on upon work supported by the National Science
Foundation under Grant Nos. 1543656 and 1954372 and by the National Institutes
of Health under Grant Nos. R01EB009675 and R01AG065218.


# Required packages

This program requires `numpy`, `scikit-learn`, and `joblib`.
Testing has been done with versions `numpy==1.19.2`, `scikit-learn==0.23.2`, and
`joblib==0.17.0`.


# Running AL

AL is run using the following command-line format (requires Python 3):
```
python al.py <options>+ <files>+
```
The options, input file format, and output are described below.


# Options

The following AL options are currently available.

`--mode <mode_value>`

AL supports several modes of operation. The default mode is `TRAIN`.

*  `TRAIN`: In this mode, AL will train a classifier from the provided labeled data
  and will store the model as a pkl file in the model_path directory.

*  `TEST`: In this mode, AL will load a pretrained model from the model_path
  directory and will test the performance of the model on new data.

*  `CV`: In this mode, AL will perform 3-fold cross validation on a specified
  datafile.

*  `PARTITION`: In this mode, AL performs a variant of testing in which it trains
  a model on the first 2/3 of data and tests on the last 1/3 of data. This
  avoids potential data overlap that occurs in standard cross-validation
  testing.

*  `ANNOTATE`: In this mode, AL will load a pretrained model from the model_path
  directory and use it to add activity labels to an input set of data.

*  `WRITE`: In this mode, AL extracts features from the input set of data that
  would normally be used for training/testing and writes the features out
  in a csv file.

*  `LOO`: In this mode, AL expects multiple data files to be input (one per
  home/subject) and performs leave-one-subject-out testing.

     AL assumes that each subject is represented by a separate data file.
  In the `LOO` mode only, files can be listed individually at the end of the
  command line. For example,
  `python al.py --mode LOO subject1 subject2 subject3 subject4 subject5`
  will perform leave-one-out testing for files `subject1..subject5` while
  also ignoring the Other_Activity category.


`--ignoreother`

This option ignores all sensor events affiliated with the activity
Other_Activity.  The default is True.


`--clusterother`

This option divides the Other_Activity category into subclasses using k-means
clustering. Because the Other_Activity category is often much larger than the
others, this can improve class balance and overall predictive performance.
The default is True.


`--data <data_file>`

Specification of the data file to use for training, testing, cross-validation,
partition, writing as a csv file of features, or annotating.  This is not an
appropriate option to use for LOO mode. The default value is "data".


`--model <model_file>`

Specification of the model file to use for training, testing, cross-validation,
partition, writing as a csv file of features, or annotating.  This is not an
appropriate option to use for LOO mode. The default value is "model".


`--activities <list>`

Specify list of activities to use for training, testing, or annotating data.
be at least one occurrence of each activity in the training data. The correct
format of this mode is
`--activities "[Bed_Toilet_Transition,Go_To_Sleep,Sleep,Wake_Up,Personal_Hygiene,Other_Activity,Bathe,Step_Out,Morning_Meds,Cook_Breakfast,Eat_Breakfast,Wash_Breakfast_Dishes,Dress,Toilet,Exercise,Watch_TV,Leave_Home,Enter_Home,Work_At_Table,Evening_Meds,Cook_Lunch,Eat_Lunch,Wash_Lunch_Dishes,Relax,Cook_Dinner,Eat_Dinner,Wash_Dinner_Dishes,Read,Sleep_Out_Of_Bed,Phone,Wash_Dishes,Cook,Eat]"`
which is also the default value for the list of activities.


`--sensors <list>`

Specify list of sensors that appear in the data file. The correct format of this
mode is
`--sensors "[Bath,Bathroom,BathroomDoor,BathroomLight,Bed,Bedroom,BedroomDoor,BedroomLight,Chair,Closet,ClosetDoor,ClosetLight,Couch,Desk,DiningRoom,DiningRoomLight,Entry,FrontDoor,FrontDoorLight,GarageDoor,GuestBathroom,GuestRoom,Hall,HallLight,Kitchen,KitchenCabinet,KitchenLight,LaundryRoom,LivingRoom,LivingRoomLight,LoungeChair,MedCabinet,Office,OfficeLight,OtherRoom,OutsideDoor,Refrigerator,Sink,Stove,TV,Toilet,WorkArea,WorkAreaLight]"`
which is also the default value for the list of sensors.


# Input File(s)

The input file(s) contains time-stamped sensor readings. An example is provided
in the file "data". Each line of the input file contains a reading for
a single sensor.  An example is shown below.
```
2011-06-15 01:40:48.900973 Bedroom Bedroom ON Sleep
2011-06-15 01:40:50.015327 Bedroom Bedroom OFF Sleep
2011-06-15 01:41:27.905913 Bedroom Bedroom ON Wake_Up
2011-06-15 01:41:29.789869 Bedroom Bedroom OFF Wake_Up
2011-06-15 01:41:30.712847 Bedroom Bedroom ON Wake_Up
2011-06-15 01:41:31.842606 Bedroom Bedroom OFF Wake_Up
2011-06-15 01:41:40.205759 Bathroom Bathroom ON Bed_Toilet_Transition
2011-06-15 01:41:41.296767 Bathroom Bathroom OFF Bed_Toilet_Transition
2011-06-15 01:41:43.385244 Bathroom Bathroom ON Bed_Toilet_Transition
2011-06-15 01:41:44.454793 Bathroom Bathroom OFF Bed_Toilet_Transition
2011-06-15 01:41:47.303222 Bathroom Bathroom ON Bed_Toilet_Transition
2011-06-15 01:41:48.980746 Bathroom Bathroom OFF Bed_Toilet_Transition
```
The general format for the data contains 6 fields per line. The fields are:

* date: yyyy-mm-dd
* time: hh:mm:ss.ms
* sensor name
* sensor name (in the current version the same name appears twice)
* sensor reading
* label: this is a string indicating the activity label

As can be seen from the example, sensors are not sampled at a constant rate.
Instead, they send a message when there is a change in the state of the sensor
(e.g., ON or OFF for motion sensors, OPEN or CLOSED for door sensors, large
change in value for light or temperature sensors).


# Features

AL extracts a vector of features for each non-overlapping sensor sequence of
length max_window (max_window indicates the number of distinct sensor readings
that are included in the sequence). A random forest classifier maps the feature
vector to the activity label(s). Current extracted features are:

*  time of the last sensor event in window (hour)
*  time of the last sensor event in window (seconds)
*  day of the week for the last sensor event in window
*  time duration of entire window
*  time elapsed since previous sensor event
*  dominant sensor (sensor firing most often) for previous window
*  dominant sensor two windows back
*  last sensor event in current window
*  last sensor location in current window
*  last motion sensor location in current window
*  complexity of window (entropy calculated from sensor counts)
*  change in activity level between two halves of current window
*  number of transitions between areas in current window
*  number of distinct sensors in current window
*  counts for each sensor in current window
*  time elasped since each sensor last fired
