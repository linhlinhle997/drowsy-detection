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
Download the [Eye dataset](http://parnec.nuaa.edu.cn/xtan/data/datasets/dataset_B_Eye_Images.rar) and unzip to the ```bash ./data/ ``` folder. We will have four folders:
* closedLeftEyes
* closedRightEyes
* openLeftEyes
* openRightEyes

Download [this file](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) and place into the ``` ./input/ ``` folder. Rename it with trained_data.dat

## Files included:
``` trainModel.ipynb ``` : Preprocess the data by converting the images to grayscale and dividing them into training and testing sets. Train a CNN based on the training data.

Weights and json are automatically downloaded, and place into the ``` ./trained_model/ ```

## Testing Real-time with Camera
``` bash python drowsy.py ```

Press Q to quit the application

## Result
* Video saved at ``` ./output/output_video/ ```
* Text saved at ``` ./output/output_tet/ ```
