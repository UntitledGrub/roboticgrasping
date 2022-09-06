# A transformer based network for antipodal grasp detection
The TCT model is a network consisting of Swin transformer (see https://arxiv.org/abs/2103.14030) and convolutional layers to predict the quality and grasp pose of a two fingered robotic gripper for each pixel in an input image. The Swin transformer learns to detect the grasp quality score of each pixel. The Hadamard product of the original input and the grasp quality map is then computed and this modified version of the input is passed to the convolutional layers which then predict the opening width and angle of orientation. The idea is that modifying the input using the quality scores will enable the network to 'ignore' areas of low grasp quality when predicting the opening width and angle. 

A fully convolutional model with the same structure (predicting the quality map first and using that information to inform pose prediction) is also implemented as a point of comparison with the transformer based architecture. An ablation model is included which removes the convolutional layers from the TCT model and does not use the per pixel grasp quality scores to modify the original input before predicting the gripper opening width and angle of orientation.

The main components are two top level scripts, train.py and eval.py, which are interacted with via the command line.

## Setup

Download and extract the Cornell (https://www.kaggle.com/datasets/oneoneliu/cornell-grasp) and Jacquard (https://jacquard.liris.cnrs.fr/index.php) datasets. 

Run the script generate_cornell_depth.py from the command line with the argument '--path <cornell_dataset_directory>' appended.

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

## Training


## Evaluation

## Pretrained Models
Along with the Python code you I have uploaded the best performing versions of the TCT model, the fully convolutional model and the model for the ablation study. 

You can use these models and the eval.py script to reproduce my results. Note that the models trained on the Jacquard dataset are trained using a 0.95 training/test split, so you need to add '--split 0.95' when running evaluation on the models trained on the Jacquard dataset. Running the eval.py script with the following arguments to reproduce my results:

--dataset jacquard --dataset-path datasets\Jacquard_Dataset --split 0.95 --iou-eval
--dataset cornell --dataset-path datasets\Cornell_dataset --iou-eval

## Results
