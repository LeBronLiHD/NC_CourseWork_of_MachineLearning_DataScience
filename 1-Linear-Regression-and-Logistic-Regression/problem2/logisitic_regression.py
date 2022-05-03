import numpy as np
import math


def sigmoid(x):
    # the sigmoid function
    pass


class LogisticReg(object):
    def __init__(self, indim=1):
        # initialize the parameters with all zeros
        # w: shape of [d+1, 1]
        # pass
        self.w = np.zeros(shape=(indim + 1, 1))
    
    def set_param(self, weights, bias):
        # helper function to set the parameters
        # NOTE: you need to implement this to pass the autograde.
        # weights: vector of shape [d, ]
        # bias: scaler
        self.weights = weights
        self.bias = bias
        for i in range(len(weights)):
            self.w[i][0] = weights[i]
        self.w[len(weights)][0] = bias
        # print("self.w", self.w)
        # pass

    def sigmod(self, x):
        return 1.0/(1 + math.exp(-x))
    
    def get_param(self):
        # helper function to return the parameters
        # NOTE: you need to implement this to pass the autograde.
        # returns:
            # weights: vector of shape [d, ]
            # bias: scaler
        # pass
        return self.weights, self.bias

    def compute_loss(self, X, t):
        # compute the loss
        # X: feature matrix of shape [N, d]
        # t: input label of shape [N, ]
        # NOTE: return the average of the log-likelihood, NOT the sum.

        # extend the input matrix

        NEW_X = np.ones(shape=(X.shape[0], X.shape[1] + 1))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                NEW_X[i][j] = X[i][j]

        print("\nlen(t) ->", len(t))
        print("t.type", type(t))
        print("X.type", type(X))
        print("t.shape ->", t.shape)
        print("X.shape ->", X.shape)
        print("NEW_X.shape ->", NEW_X.shape)
        print("calc ->", np.dot(NEW_X, self.w))
        Y = np.dot(NEW_X, self.w)
        print("sigmod ->", self.sigmod(Y[0]))

        loss = 0
        for i in range(len(t)):
            loss += -math.log(self.sigmod(t[i] * Y[i]))

        print("loss ->", loss)
        print("loss/len(t) ->", loss/len(t))
        # compute the loss and return the loss
        return loss/len(t)


    def compute_grad(self, X, t):
        # X: feature matrix of shape [N, d]
        # grad: shape of [d, 1]
        # NOTE: return the average gradient, NOT the sum.
        NEW_X = np.ones(shape=(X.shape[0], X.shape[1] + 1))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                NEW_X[i][j] = X[i][j]
        # pass
        Y = np.dot(NEW_X, self.w)
        grad = np.zeros(shape=(NEW_X.shape[1], ))
        for i in range(NEW_X.shape[1]):
            for j in range(len(t)):
                grad[i] += NEW_X[j][i]*(t[j] * self.sigmod(t[j] * Y[j]) - t[j])
            grad[i] /= len(t)
        print("computed grad ->", grad)
        # print("NEW_X ->", NEW_X)
        return grad


    def update(self, grad, lr=0.001):
        # update the weights
        # by the gradient descent rule
        # self.w -= grad * lr
        w, b = self.get_param()
        w -= grad[:3] * lr
        b -= grad[3] * lr
        self.set_param(w, b)
        # pass


    def fit(self, X, t, lr=0.001, max_iters=1000, eps=1e-7):
        # implement the .fit() using the gradient descent method.
        # args:
        #   X: input feature matrix of shape [N, d]
        #   t: input label of shape [N, ]
        #   lr: learning rate
        #   max_iters: maximum number of iterations
        #   eps: tolerance of the loss difference 
        # TO NOTE: 
        #   extend the input features before fitting to it.
        #   return the weight matrix of shape [indim+1, 1]

        loss = 1e10
        for epoch in range(max_iters):
            # compute the loss 
            new_loss = self.compute_loss(X, t)

            # compute the gradient
            grad = self.compute_grad(X, t)

            # update the weight
            self.update(grad, lr=lr)

            # decide whether to break the loop
            if np.abs(new_loss - loss) < eps:
                return self.w


    def predict_prob(self, X):
        # implement the .predict_prob() using the parameters learned by .fit()
        # X: input feature matrix of shape [N, d]
        #   NOTE: make sure you extend the feature matrix first,
        #   the same way as what you did in .fit() method.
        # returns the prediction (likelihood) of shape [N, ]
        NEW_X = np.ones(shape=(X.shape[0], X.shape[1] + 1))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                NEW_X[i][j] = X[i][j]
        RET = np.zeros(shape=(len(X), ))
        Y = np.dot(NEW_X, self.w)
        for i in range(len(X)):
            RET[i] = self.sigmod(Y[i])
        return RET
        # pass

    def predict(self, X, threshold=0.5):
        # implement the .predict() using the .predict_prob() method
        # X: input feature matrix of shape [N, d]
        # returns the prediction of shape [N, ], where each element is -1 or 1.
        # if the probability p>threshold, we determine t=1, otherwise t=-1
        # pass
        RET = self.predict_prob(X)
        print("RET.shape ->", RET.shape)
        T = np.zeros(shape=RET.shape)
        for i in range(len(T)):
            T[i] = 1 if RET[i] > threshold else -1
        return T
