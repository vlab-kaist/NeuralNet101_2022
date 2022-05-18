# MNIST

## Direction

### Download dependencies

You should first download dependencies first : numpy and torchvision.

Copy two files to local : grader.py and mnist.py.

### Implement

You need to implement three class implements NeuronAbstract : Softmax, Sigmoid, and Affine,
one class implements ModelAbstract : Model,
and one function that returns Model that predicts MNIST.

Softmax, Sigmoid, Affine are implementation of Neuron that has forward and backward functions.
Model is implementation of ModelAbstract that has forward and backward functions.

You can test your code by running ```grader.py```. Note that you need to run grader.py in the same directory as mnist.py.

### Submit

You only need to submit mnist.py.

You may edit grader.py if you want to change the test cases, but it will not be affect to your submission.
