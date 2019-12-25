# drowsy-detection

Drowsy detection using Keras and convolution neural networks.

## Requirement
* tensorflow-gpu
* keras
* opencv-contrib-python
* dlib
* matplotlib
* numpy
* pygame
* times

## Datasets:
Download the [Eye dataset](http://parnec.nuaa.edu.cn/xtan/data/datasets/dataset_B_Eye_Images.rar) and unzip to the ``` ./data/ ``` folder. We will have four folders:
* closedLeftEyes
* closedRightEyes
* openLeftEyes
* openRightEyes

Download [this file](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) and place into the ``` ./input/ ``` folder. Rename it with ```trained_data.dat```

## Files included:
``` trainModel.ipynb ``` : Preprocess the data by converting the images to grayscale and dividing them into training and testing sets. Train a CNN based on the training data.

Weights and json save at ``` ./trained_model/ ```

## Testing Real-time with Camera
* Displayed: ```python drowsy.py ``` 
* Not displayed: ```python drowsy.py -d 0```

Press Q to quit the application

## Result
* Video saved at ``` ./output/output_video/ ``` (Only save videos when drowsiness is detected)
* Text saved at ``` ./output/output_tet/ ``` (Save current time, eye status, number of faces, paths of video if detected drowsiness)
