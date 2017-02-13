from nn import *
import numpy as np

## Add Node
x, y = Input(), Input()

f = Add(x, y)

output = forward_pass(f, topological_sort({x: 5, y: 5}))

assert output == 10, "Output %d is different than expected" % output
print("Add node test passed")


## Linear Node with 1-d inputs
features, weights, bias = Input(), Input(), Input()
linear = Linear(features, weights, bias)
feed_dict = {features: np.array((1, 2, 3)), weights: np.array((2,3,4)), bias: 3}
output = forward_pass(linear, topological_sort(feed_dict))
assert output == 23, "Output %d is different than expected" % output
print("Linear node 1-d inputs test passed")

## Linear Node with 2-d inputs
features, weights, bias = Input(), Input(), Input()
linear = Linear(features, weights, bias)
feed_dict = {features: np.array([[-1., -2.], [-1, -2]]), weights: np.array([[2., -3], [2., -3]]),
             bias: np.array([-3., -5])}
output = forward_pass(linear, topological_sort(feed_dict))
assert np.array_equal(output, np.array([[-9., 4.], [-9., 4.]])) == True, "Output %d is different than expected" % output
print("Linear node 2-d inputs test passed")

## Linear + Sigmoid with 2-d inputs
features, weights, bias = Input(), Input(), Input()
linear = Linear(features, weights, bias)
sigmoid = Sigmoid(linear)
feed_dict = {features: np.array([[-1., -2.], [-1, -2]]), weights: np.array([[2., -3], [2., -3]]),
             bias: np.array([-3., -5])}
output = forward_pass(sigmoid, topological_sort(feed_dict))
expected_output = np.array([[1.23394576e-04, 9.82013790e-01], [1.23394576e-04, 9.82013790e-01]])
np.testing.assert_array_almost_equal(output, expected_output) == True
print("Linear + Sigmoid node 2-d inputs test passed")





