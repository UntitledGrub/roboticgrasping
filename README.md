# A transformer based network for antipodal grasp detection
The TCT model is a network consisting of a Swin transformer (see https://arxiv.org/abs/2103.14030) and convolutional layers to predict the quality and grasp pose of a two fingered robotic gripper for each pixel in an input image. The Swin transformer learns to detect the grasp quality score of each pixel. The Hadamard product of the original input and the grasp quality map is then computed and this modified version of the input is passed to the convolutional layers which then predict the opening width and angle of orientation. The idea is that modifying the input using the quality scores will enable the network to 'ignore' areas of low grasp quality when predicting the opening width and angle. 

A fully convolutional model with the same structure (predicting the quality map first and using that information to inform pose prediction) is also implemented as a point of comparison with the transformer based architecture. An ablation model is included which removes the convolutional layers from the TCT model and does not use the per pixel grasp quality scores to modify the original input before predicting the gripper opening width and angle of orientation.

The main components are two top level scripts, train.py and eval.py, which are interacted with via the command line.

## Setup

Download and extract the Cornell (https://www.kaggle.com/datasets/oneoneliu/cornell-grasp) and Jacquard (https://jacquard.liris.cnrs.fr/index.php) datasets. Access to the Jacquard dataset must be requested from the team at Ecole Centrale de Lyon who created it, and it is used in this work with their permission. The Cornell dataset is available publicly.

Before training or evaluating a model, run the script generate_cornell_depth.py from the command line with the argument '--path <cornell_dataset_directory>'.

## Dependencies

torch <br />
cuda 11.3 <br />
numpy <br />
matplotlib <br />
opencv-python <br />
torchsummary <br />
imageio <br />
sklearn <br />
scikit-image <br />
timm

## Training


## Evaluation

## Pretrained Models
Along with the Python code I have uploaded the best performing versions of the TCT model, the fully convolutional model and the model for the ablation study.

To use the pretrained models, download them from [here](https://drive.google.com/file/d/1l5RKy4Y8sDSC9-BvUsDuEeIukYSJ4pj1/view?usp=sharing). 

You can use these models and the eval.py script to reproduce my results. Note that the models trained on the Jacquard dataset are trained using a 0.95 training/test split, so you need to add '--split 0.95' when running evaluation on the models trained on the Jacquard dataset. Run the eval.py script with the following commands to reproduce my results:

eval.py --network 'path_to_model' --dataset jacquard --dataset-path datasets\Jacquard_Dataset --split 0.95 --iou-eval <br />
eval.py --network 'path_to_model' --dataset cornell --dataset-path datasets\Cornell_dataset --iou-eval

## Results

![image](https://user-images.githubusercontent.com/34168073/191499512-9e643fb7-a11b-400b-858d-b205e6c00b27.png)
Example of the trained model's output. The image in the top left of the figure is part of the validation dataset, and was unseen during training. The model is able to filter out irrelevant regions of the image as areas of the quality map outside the boundaries of the object pictured have a grasp quality score of 0. The grasp quality map approximately follows the shape of the object. Likewise, the model’s width and grasp predictions modulate around the region the pictured object occupies with the background of the image being ignored.

![image](https://user-images.githubusercontent.com/34168073/191499725-c57ad855-4ecf-460f-99bc-aef68ba8b1e4.png)
Comparison of TCT network performance with state of the art pixel-wise grasp detection models *Reported inference speeds in these studies were measured on different hardware so comparison is not possible


## Acknowledgement

Dataset processing, training and evaluation code inspired and modified from https://github.com/dougsm/ggcnn. 

The Swin transformer code used here is an adapted version of the original Swin transformer code which can be found here: https://github.com/microsoft/Swin-Transformer.
