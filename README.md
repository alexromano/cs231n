## Stanford CS231n: Convolutional Neural Networks for Visual Recognition
Solutions for CS231n Assignments
### assignment1:
- [kNN Classifier](assignment1/knn.ipynb)
- [SVM Classifier](assignment1/svm.ipynb)
- [Softmax Classifier](assignment1/softmax.ipynb)
- [Two Layer Neural Net](assignment1/two_layer_net.ipynb) (network implementation in [neural_net.py](assignment1/cs231n/classifiers/neural_net.py))
- [Feature Representations](assignment1/features.ipynb): experiment with higher level representations (Histogram of Gradients)
### assignment2:
- [Fully Connected Nets](assignment2/FullyConnectedNets.ipynb): Modular layer design in [layers.py](assignment2/cs231n/layers.py) and [fc_net.py](assignment2/cs231n/classifiers/fc_net.py) 
- [BatchNorm](assignment2/BatchNormalization.ipynb)
- [Dropout](assignment2/Dropout.ipynb)
- [CNN](assignment2/ConvolutionalNetworks.ipynb): CNN, Max Pooling, Spatial Batchnorm, GroupNorm implementations

### assignment3:
- [RNN_Captioning](assignment3/RNN_Captioning.ipynb): implements a Vanilla RNN for image captioning MS-COCO dataset. Layer implementations are in [rnn_layers](assignment3/cs231n/rnn_layers.py)
- [LSTM_Captioning](assignment3/LSTM_Captioning.ipynb): implements a LSTM for the same task
- [Network Visualization](assignment3/NetworkVisualization-TensorFlow.ipynb): uses a pretrained SqueezeNet model to compute gradients with respect to images and product saliency maps and fooling images in Tensorflow
- [Style Transfer](assignment3/StyleTransfer-TensorFlow.ipynb): implements style transfer in Tensorflow
- [ Generative Adversarial Networks](assignment3/GANs-TensorFlow.ipynb): implements vanilla GAN, Least Squares GAN, and DCGAN in Tensorflow
