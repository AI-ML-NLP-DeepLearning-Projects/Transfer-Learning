# Transfer-Learning
Explored two different types of Transfer Learning.

 The first approach is using a pre-trained CNN as a fixed feature extractor. In this technique,
all layers of the CNN are frozen except for the last fully-connected layer. This last
layer is changed to suit the task at hand.Took a pretrained model and replace the last fully connected layer which classifies
images into 1000 classes into a new classifier that is adapted to classify images
into 5 classes.
 In the second approach, once again, replaced the final fully connected layer of
the network with a 5-class classifier. However, this time trained the last layer
for CIFAR-10 dataset.
 In both cases, Took a small 8,000 image dataset comprising 5 classes and
train the CNN on this small dataset.
 Use Resnet as pre-trained model.

# Keras # Cifar-10 # CNN
