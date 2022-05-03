from difflib import unified_diff
import unittest
import numpy as np
import logisitic_regression as lr
from numpy.testing import assert_allclose
import pickle as pkl

seed = 10417617

with open("tests.pkl","rb") as f: 
    tests = pkl.load(f)

TOLERANCE = 1e-5

# to run one test: python -m unittest tests.SetGet
# to run all tests: python -m unittest tests


class TestSetGet(unittest.TestCase):
	def test(self):
		weights, bias = np.random.uniform(-1, 1, size=(3,)), np.random.uniform(-1,1)

		model = lr.LogisticReg(indim=3)
		model.set_param(weights, bias)
		w, b = model.get_param()

		assert_allclose(weights, w, atol=TOLERANCE)
		assert_allclose(bias, b, atol=TOLERANCE)


class TestLoss(unittest.TestCase):
	def test(self):
		X, t, weights, bias, loss_true = tests[0]
		lr_model = lr.LogisticReg(indim=3)
		lr_model.set_param(weights=weights, bias=bias)
		loss = lr_model.compute_loss(X, t)
		assert_allclose(loss, loss_true, atol=TOLERANCE)


class TestGrad(unittest.TestCase):
	def test(self):
		X, t, weights, bias, grad_true = tests[1]
		print("grad_true ->", grad_true)
		print(type(grad_true))
		print(grad_true.shape)
		lr_model = lr.LogisticReg(indim=3)
		lr_model.set_param(weights=weights, bias=bias)
		grad = lr_model.compute_grad(X, t).squeeze()
		assert_allclose(sorted(grad_true), sorted(grad), atol=TOLERANCE)


class TestUpdate(unittest.TestCase):
	def test(self):
		X, t, weights, bias, w_after, b_after = tests[2]
		lr_model = lr.LogisticReg(indim=3)
		lr_model.set_param(weights=weights, bias=bias)
		grad = lr_model.compute_grad(X, t)
		lr_model.update(grad, lr=0.001)
		w, b = lr_model.get_param()
		assert_allclose(w, w_after, atol=TOLERANCE)
		assert_allclose(b, b_after, atol=TOLERANCE)


class TestPredict(unittest.TestCase):
	def test(self):
		X, t, weights, bias, predictions = tests[3]
		lr_model = lr.LogisticReg(indim=2)
		lr_model.set_param(weights=weights, bias=bias)
		
		p = lr_model.predict(X)

		assert_allclose(predictions, p, atol=TOLERANCE)

test = TestSetGet()
test.test()
