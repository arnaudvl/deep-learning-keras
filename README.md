# deep-learning-keras
Easily create, train and make predictions with CNN and feedforward NN models using keras in python.

## Getting started

### Prerequisites

Make sure you have up-to-date versions installed of:

  - keras
  - numpy
  - pandas
  - scikit-learn
  - scipy

### Installation

Clone the repository in your local workspace:

```
git clone https://github.com/arnaudvl/deep-learning-keras
```

## Functionality

There are 3 main modules in dlkeras: cnn, cnn_transfer_learning and dnn.

Each of them allow you to create and train neural networks, make predictions,
print the training history, save models, run cross-validation and make out-of-fold predictions in a few lines of code.

cnn and cnn_transfer_learning deal with convolutional neural nets. They also facilitate on the fly image augmentation, allow for additional non-image input data, and apply image scaling. cnn_transfer_learning allows transfer learning from a range of base models, can find the optimal layer to truncate the base model from and fine tunes the base model layers.

dnn is made for feedforward deep neural nets and has convenient ways to scale the input data and apply one-hot-encoding.

The functionality is illustrated in the examples.
