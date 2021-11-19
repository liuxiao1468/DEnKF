# IRL-Depth_for_walking
This is the repo for the paper "An Environment-aware Predictive Modeling Framework for Human-Robot Symbiotic Walking"

About the network structure:

The depth prediction network is built upon an dense autoencoder with skip connection from encoder to decoder.
The bottlenect has size 64
The building blocks include regular conv layers (with kernel 3x3 and 1x1), conv_blocks and identity_blocks (from Resnet).
Loss function: MSE + gradient + Total variance (TV)

Input: 90x160 RGB images
Output: 90x160 depth images

We have added the Docker implementation for training the model.

Full version of the code will be released with the accepted version of the paper.
