import nn
import numpy as np

class PerceptronModel(object):
    def __init__(self, dim):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dim` is the dimensionality of the data.
        For example, dim=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dim)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x_point):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x_point: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x_point)

    def get_prediction(self, x_point):
        """
        Calculates the predicted class for a single data point `x_point`.

        Returns: -1 or 1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(nn.DotProduct(self.w, x_point)) >= 0:
            return 1
        # dot product is negative
        else:
            return -1

    def train_model(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        keep = True
        batch_size = 1
        while keep:
            # terminate while done
            keep = False
            # retrieve training examples from training database
            for x, y in dataset.iterate_once(batch_size):
                # get the prediction
                prediction = self.get_prediction(x)
                # check if the prediction is the same as the correct direction y
                # if not, update the weights
                if prediction != nn.as_scalar(y):
                    self.w.update(nn.as_scalar(y), x)
                    keep = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

         # weight sizes must be batch_size X num_features
        self.weight1 = nn.Parameter(1, 20)
        self.weight2 = nn.Parameter(20, 10)
        self.weight3 = nn.Parameter(10, 1)
        # bias sizes must be 1 X num_features
        self.b1 = nn.Parameter(1, 20)
        self.b2 = nn.Parameter(1, 10)
        self.b3 = nn.Parameter(1, 1)

        self.batch_size = 10
        # learning rate
        self.lr = 0.001


    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # f(x) = reLu(reLu(x * W1 + b1) * W2 + b2) * W3 + b3
        # first ReLu
        matrix1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.b1))
        # second ReLu
        matrix2 = nn.ReLU(nn.AddBias(nn.Linear(matrix1, self.weight2), self.b2))
        # prediction
        output = nn.AddBias(nn.Linear(matrix2, self.weight3), self.b3)
        return output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"

        return nn.SquareLoss(self.run(x), y) 

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        keep = True
        while keep:
            # retrieve training examples from training database
            for x, y in dataset.iterate_once(self.batch_size):
                # get loss
                loss = self.get_loss(x, y)
                # get gradient vectors
                w1, bias1, w2, bias2, w3, bias3 = nn.gradients([self.weight1, self.b1, self.weight2, self.b2,
                                                                self.weight3, self.b3], loss)
                # update the weights and biases by learning rate
                self.weight1.update(-self.lr, w1)
                self.weight2.update(-self.lr, w2)
                self.weight3.update(-self.lr, w3)
                self.b1.update(-self.lr, bias1)
                self.b2.update(-self.lr, bias2)
                self.b3.update(-self.lr, bias3)
            # check the loss
            if nn.as_scalar(loss) < 0.002:  # ?: 0.02 is not working there!
                keep = False

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"

        # three-layer neutral network
        # weight sizes must be batch_size X num_features
        self.weight1 = nn.Parameter(784, 200)  # initial wright vectors: 784-dimensional vector
        self.weight2 = nn.Parameter(200, 100)
        self.weight3 = nn.Parameter(100, 10)
        # bias sizes must be 1 X num_features
        self.b1 = nn.Parameter(1, 200)
        self.b2 = nn.Parameter(1, 100)
        self.b3 = nn.Parameter(1, 10)

        self.batch_size = 20
        # learning rate
        self.lr = 0.01

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # first ReLu
        matrix1 = nn.ReLU(nn.AddBias(nn.Linear(x, self.weight1), self.b1))
        # second ReLu
        matrix2 = nn.ReLU(nn.AddBias(nn.Linear(matrix1, self.weight2), self.b2))
        # prediction
        output = nn.AddBias(nn.Linear(matrix2, self.weight3), self.b3)
        return output


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train_model(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        keep = True
        # start the training process
        while keep:
            # retrieve training examples from training database
            for x, y in dataset.iterate_once(self.batch_size):
                # get loss
                loss = self.get_loss(x, y)
                # get gradient vectors
                w1, bias1, w2, bias2, w3, bias3 = nn.gradients([self.weight1, self.b1, self.weight2, self.b2,
                                                                self.weight3, self.b3], loss)
                # update the weights and biases by learning rate
                self.weight1.update(-self.lr, w1)
                self.weight2.update(-self.lr, w2)
                self.weight3.update(-self.lr, w3)
                self.b1.update(-self.lr, bias1)
                self.b2.update(-self.lr, bias2)
                self.b3.update(-self.lr, bias3)
            # check the validation accuracy
            # terminate when the validation accuracy achieves 97%
            if dataset.get_validation_accuracy() > 0.97:
                keep = False


