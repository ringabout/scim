import arraymancer, math



proc hanning*[T: SomeFloat](n: int): Tensor[T] {.discardable.} =
  ## Creates a hanning window of length n.
  ## echo hanning[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](n)
  for i in 0 ..< n:
    result[i] = 0.5 - 0.5 * cos(2 * i / (n - 1) * Pi)



proc hamming*[T: SomeFloat](n: int): Tensor[T] {.discardable.} =
  ## Creates a hamming window of length n.
  ## echo hamming[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](n)
  for i in 0 ..< n:
    result[i] = 0.54 - 0.46 * cos(2 * i / (n - 1) * Pi)




proc rect*[T: SomeFloat](n: int): Tensor[T] =
  ## Creates a rect window of length n.
  ## echo rect[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](n)
  for i in 0 ..< n:
    result[i] = T(1)


proc bartlett*[T: SomeFloat](n: int): Tensor[T] =
  ## Creates a rect window of length n.
  ## echo rect[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](n)
  for i in 0 ..< n:
    result[i] = 1 - 2 / (n - 1) * abs((2 * i - n + 1) / 2)

proc blackman*[T: SomeFloat](n: int): Tensor[T] =
  ## Creates a rect window of length n.
  ## echo rect[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](n)
  for i in 0 ..< n:
    result[i] = 0.42 - 0.5 * cos(2 * i / (n - 1) * Pi) + 0.08 * cos(4 * i / (n - 1) * Pi)


proc kaiser*[T: SomeFloat](n: int, beta: float): Tensor[T] =

  # from numpy.dual import i0
  # if M == 1:
  #     return np.array([1.])
  # n = arange(0, M)
  # alpha = (M-1)/2.0
  # return i0(beta * sqrt(1-((n-alpha)/alpha)**2.0))/i0(float(beta))


when isMainModule:
  discard hanning[float](12)
  discard hanning[float32](1)
  discard hanning[float64](12)
  discard hamming[float](12)
  discard hamming[float32](1)
  discard hamming[float64](12)
  discard rect[float](12)
  discard rect[float32](1)
  discard rect[float64](12)
  discard bartlett[float](12)
  discard bartlett[float32](12)
  discard bartlett[float64](1)

  discard blackman[float](12)
  discard blackman[float32](12)
  discard blackman[float64](1)

  # import timeit
  # echo "test1"
  # echo timeGo(hanning[float](12000))
  # echo "test2"
  # echo timeGo(hamming[float](12000))
