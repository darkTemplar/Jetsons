from nn import *
import numpy as np

## Add Node
x, y = Input(), Input()

f = Add(x, y)

output = forward_pass(f, topological_sort({x: 5, y: 5}))

assert output == 10, "Output %d is different than expected" % output
print("Add node test passed")


## Linear Node
features, weights, bias = Input(), Input(), Input()
linear = Linear(features, weights, bias)
feed_dict = {features: np.array((1, 2, 3)), weights: np.array((2,3,4)), bias: 3}
output = forward_pass(linear, topological_sort(feed_dict))
assert output == 23, "Output %d is different than expected" % output
print("Linear node test passed")






