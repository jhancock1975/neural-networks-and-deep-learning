import numpy as np

def feed_forward(W, a):
  a_next = a
  for w_cur in W:
    a_next = sigmoid(w_cur.dot(a_next))
  print("output ", a_next)
  return a_next

def back_prop(W, a, y):
  partial_cost_wrt_a = np.subtract(a, y)
  partial_cost_wrt_a = a_next
def quad_cost(y, actual):

  diff = np.subtract(y, actual)
  print("diff = ", diff)

  elts_squared = np.square(diff)
  print("elts squared = ", elts_squared)

  sum_elts = np.sum(elts_squared)
  print("sum elements = ", sum_elts)

  return 0.5 * (sum_elts ** 0.5) 

def test_quad_cost():
  if ( abs(1.0564132315 
      - quad_cost(np.array([[1.0],[0.0]]), 
          np.array([[1.29274584], [2.09244726]]))) 
      < 0.00001 ):
   return "pass" 
  else:
    return "fail"

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

# correct output
y=np.array([[1.0],[0.0]])

# weights
W = [np.random.randn(2,3),
      np.random.randn(5,2),
      np.random.randn(2,5)]

# inputs to first layer
layer1_a = np.random.randn(3,1)

# biases
b = np.array([0.1,0.2,0.3,0.4,0.5])

output1 = feed_forward(W, layer1_a)

print(" test result: ", test_quad_cost())


