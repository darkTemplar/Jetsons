from nn import *

x, y = Input(), Input()

f = Add(x, y)

output = forward_pass(f, topological_sort({x: 5, y: 5}))

print(output)
