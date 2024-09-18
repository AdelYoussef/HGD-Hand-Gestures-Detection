# HGD (Hand Gestures Detection)

![image](https://user-images.githubusercontent.com/63298005/179505293-8c0d2cf1-2fc6-48f8-87dd-09acb4f5f15d.png)



# Training
The model was trained using Hand Gestures dataset or senz3d dataset which can be found [here](https://lttm.dei.unipd.it/downloads/gesture/) the training was done using the TensorFlow object detection API [can be found here ](https://github.com/tensorflow/models/tree/master/research/object_detection)

## First 
The data needs to be labeled, this step could be done using [LabelImg](https://github.com/heartexlabs/labelImg)
LabelImg outputs a boundry box and a label for each image and saves data in an xml file 


## Second 
the output xml files needes to be converted to csv files, this could be done using the provided xml_csv.py script found in "needed scripts" folder [Source](https://github.com/datitran/raccoon_dataset).
These csv files are the used to generate tf.record files, tf.record files are the files which are fed to the model along with training and testing images. tf.record files are generated using ge_tf_record.py script found in "needed scripts" folder [Source](https://github.com/datitran/raccoon_dataset). Don't forget to change "class_text_to_int" labels to match your training labels.

## Third
Now the finals step left is to run the model with the training dataset. This could be done using.

### Training
This command runs the model training script.

`python D:/adel/deep_learning/models/research/object_detection/model_main_tf2.py --model_dir=D:/adel/deep_learning/hand_poses/mobilenet_v2 --pipeline_config_path=D:/adel/deep_learning/hand_poses/mobilenet_v2/pipeline.config --num_train_steps=2000`



### Save Model
This command exports the trained model to the desired directory.

`python D:/adel/deep_learning/models/research/object_detection/exporter_main_v2.py  --input_type=image_tensor --pipeline_config_path=D:/adel/deep_learning/hand_poses/mobilenet_v2/pipeline.config --trained_checkpoint_dir=D:/adel/deep_learning/hand_poses/mobilenet_v2 --output_directory=D:/adel/deep_learning/hand_poses/mobilenet_v2/export`



### Test
Model testing could be done by running a detection script wether for a single image or for a video feed. Also could be found in "needed scripts" folder. Just copy the detection scripts in the research dirctory found in the downloaded TensorFlow object detection API folder.
