# plant-disease-detection
This project deals with the development of a basis for an artificial intelligence-based  system for monitoring the condition of the cannabis plant.
The goal of the system is to provide a real-time for identifying diseases/deficiencies in cannabis plants.
At the core of the system are neural-network-based models, that allow monitoring on the condition of the plant using an image.

![image](https://user-images.githubusercontent.com/97550175/183244613-cdfc7e74-6a9e-4309-a75c-6d7be75ad3e7.png)


![image](https://user-images.githubusercontent.com/97550175/183244679-1b1608d1-e9e9-4199-bd96-08af3406ded4.png)


Examples from the streamlit gui app:

![image](https://user-images.githubusercontent.com/97550175/183244795-addd54d6-ea60-441d-867e-03691d196e83.png)


![image](https://user-images.githubusercontent.com/97550175/183244802-272ccd2e-6b48-4236-9aff-746c5af597e4.png)


The system performs the initial prediction, whether there is a plant in the image, using a model based on Resnet101 neural networks that was trained at an earlier stage.
The model architecture:

![image](https://user-images.githubusercontent.com/97550175/183244866-900b9ff8-a553-48e8-9385-e64d28440e2a.png)


Resnet101 is a Convolutional Neural Network containing 101 layers. The size of the input that this network receives is 224x244.
The uniqueness of this network is in solving the problem of "vanishing gradients" of networks with a large number of layers with residuals.
The role of this network in the system is a binary classification of the input image as to whether there is a plant in the image.
This is an initial filtering of images that do not contain plants so that we do not perform prediction on these images and in addition so that images that are not plants are not saved in the database.


The system performs the prediction of the disease/deficiency using a Vit 16 B neural network model based on Self-Attention that was trained at an earlier stage.
The model architecture:

![image](https://user-images.githubusercontent.com/97550175/183245014-e47ace82-5c15-4194-b650-3ea41f3a05e8.png)


Vit 16 B (Vision Transformer) is a classification network that adopts a Self-Attention mechanism.
This model divides the image into patches of size 16x16 and performs classification by differential weighting of each part of the input image.
At its core, the model operates on the nature of the operation of the transformer used in the field of natural language processing.
The learning in this network is expressed by measuring the relationship between the -Patches carried out with the help of the Self-Attention mechanism.
The role of this network in the system is to classify the input image into one of 7 different categories.
This is the central model on which the system is based.
The images and output of this model are saved in the database for further iterations to improve the performance of the model.

Dataset:

![image](https://user-images.githubusercontent.com/97550175/183245146-33f2dc09-2e17-4d73-b66b-f6713ebd5368.png)

![image](https://user-images.githubusercontent.com/97550175/183245151-e6338b95-f9ba-47af-9b78-13d35fb4ccbd.png)

Confusion Matrix:

![image](https://user-images.githubusercontent.com/97550175/183245168-be1ad838-512c-4d6b-86fb-6518bfbd7d8b.png)
