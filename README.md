# A transformer based network for antipodal grasp detection
The TCT model is a network consisting of Swin transformer (see https://arxiv.org/abs/2103.14030) and convolutional layers to predict the quality and grasp pose of a two fingered robotic gripper for each pixel in an input image. The Swin transformer learns to detect the grasp quality score of each pixel. The Hadamard product of the original input and the grasp quality map is then computed and this modified version of the input is passed to the convolutional layers which then predict the opening width and angle of orientation. The idea is that modifying the input using the quality scores will enable the network to 'ignore' areas of low grasp quality when predicting the opening width and angle. 

A fully convolutional model with the same structure (predicting the quality map first and using that information to inform pose prediction) is also implemented as a point of comparison with the transformer based architecture. An ablation model is included which removes the convolutional layers from the TCT model and does not use the per pixel grasp quality scores to modify the original input before predicting the gripper opening width and angle of orientation. 

## Dependencies

torch <br />
numpy <br />
matplotlib <br />
opencv-python <br />
torchsummary <br />
imageio <br />
sklearn <br />
scikit-image <br />
timm

## Setup

Download and extract the Cornell (https://www.kaggle.com/datasets/oneoneliu/cornell-grasp) and Jacquard (https://jacquard.liris.cnrs.fr/index.php) datasets. 

Run the script generate_cornell_depth.py from the command line with the argument '--path <cornell_dataset_directory>' appended.

## Training

## Evaluation

## Pretrained Models

## Results
