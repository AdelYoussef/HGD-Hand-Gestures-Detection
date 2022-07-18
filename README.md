# TensorFlow-obj_det-API

![image](https://user-images.githubusercontent.com/63298005/179505293-8c0d2cf1-2fc6-48f8-87dd-09acb4f5f15d.png)



# Training
The model was trained using Hand Gestures dataset or senz3d dataset which can be found [here](https://lttm.dei.unipd.it/downloads/gesture/) the training was done using the TensorFlow object detection API [can be found here ](https://github.com/tensorflow/models/tree/master/research/object_detection)

## First 
The data needs to be labeled, this step could be done using [LabelImg](https://github.com/heartexlabs/labelImg)
LabelImg outputs a boundry box and a label for each image and saves data in an xml file 


## Second 
the output xml files needes to be converted to csv files, this could be done using the provided xml_csv.py script [Source](https://github.com/datitran/raccoon_dataset).
These csv files are the used to generate tf.record files, tf.record files are the files which are fed to the model along with training and testing images.
