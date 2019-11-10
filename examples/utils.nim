import arraymancer

var
  rows = 12345
  x = randomTensor(rows, max=5)
  classes = 6
  y = newTensor[int](rows, classes)


for i in 0 ..< rows:
  y[i, x[i]] = 1

echo y
# proc toCategorical*[T](data: Tensor[T], num_classes: int): Tensor[T] {.noinit} = 
#   discard
