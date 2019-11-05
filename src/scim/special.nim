import arraymancer, math, complex, fenv

type
  RankError* = Exception



proc j0*(x: cdouble): cdouble {.importc, header: "math.h".}
proc j1*(x: cdouble): cdouble {.importc, header: "math.h".}
proc jn*(n: cint, x: cdouble): cdouble {.importc, header: "math.h".}
proc y0*(x: cdouble): cdouble {.importc, header: "math.h".}
proc y1*(x: cdouble): cdouble {.importc, header: "math.h".}
proc yn*(n: cint, x: cdouble): cdouble {.importc, header: "math.h".}



proc angle*[T: SomeFloat](z: Complex[T], degrees: bool = false): T =
  result = arctan2(z.im, z.re)
  if degrees:
    result *= 180 / Pi

proc angle*(t: Tensor[Complex[float64]], degrees: bool = false): Tensor[
    float64] {.noInit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(angle(x, degrees))

proc angle*(t: Tensor[Complex[float32]], degrees: bool = false): Tensor[
    float32] {.noInit.} =
  ## Return a Tensor with absolute values of all elements
  t.map_inline(angle(x, degrees))


# TODO axis
proc roll*[T](input: Tensor[T], shift: int=1, axis:int= -1): Tensor[T] = 
  if shift == 0:
    return input

  if input.rank == 1:
    let cols = input.size
    var s: int = shift
    while s < 0:
      s += cols
    s = s mod cols
    result = newTensor[T](1, cols)
    let divPoint = cols - s
    for j in 0 ..< divPoint:
      result[0, j] = input[j+s]
    for j in divPoint ..< cols:
      result[0, j] = input[j+s-cols]
  elif input.rank == 2:
    let
      rows = input.shape[0]
      cols = input.shape[1]
    var s: int = shift
    while s < 0:
      s += cols
    s = s mod cols
    result = newTensor[T](rows, cols)
    let divPoint = cols - s
    for i in 0 ..< rows:
      for j in 0 ..< divPoint:
        result[i, j] = input[i, j+s]
      for j in divPoint ..< cols:
        result[i, j] = input[i, j+s-cols]


proc roll*[T](input: var Tensor[T], shift: int=1, axis:int= -1) = 
  if shift == 0:
    return

  if input.rank == 1:
    input = input.reshape(1, input.size)
  let 
    rows = input.shape[0]
    cols = input.shape[1]
  var s: int = shift
  while s < 0:
    s += cols
  s = s mod cols
  let n = gcd(cols, s)
  for i in 0 ..< rows:
    for j in 0 ..< n:
      let temp = input[i, j]
      var k: int = j
      while true:
        var t = k + s
        if t >= cols:
          t -= cols
        if t == j:
          break
        input[i, k] = input[i, t]
        k = t
      input[i, k] = temp




proc freq2mel*(freq: float): float = 
  ## converting from frequency to Mel scale.
  ## freq: The frequency values in Hz.
  ## returns: The mel scale values.
  var nonZero = 1.0 + freq / 700.0
  if nonZero == 0:
    nonZero = epsilon(float)
  return 1127.0 * ln(nonZero)

proc freq2mel*(freq: float32): float32 = 
  ## converting from frequency to Mel scale.
  ## freq: The frequency values in Hz.
  ## returns: The mel scale values.
  var nonZero = 1.0 + freq / 700.0
  if nonZero == 0:
    nonZero = epsilon(float32)
  return 1127.0 * ln(nonZero)


proc mel2freq*(mel: float): float = 
  ## converting from Mel scale to frequency.
  ## param mel: The mel scale values.
  ## returns: The frequency values in Hz.
  return 700.0 * (exp(mel / 1127.0) - 1.0)


proc mel2freq*(mel: float32): float32 = 
  ## converting from Mel scale to frequency.
  ## param mel: The mel scale values.
  ## returns: The frequency values in Hz.
  return 700.0 * (exp(mel / 1127.0) - 1.0)


makeUniversal(j0)
makeUniversal(j1)
makeUniversal(jn)
makeUniversal(y0)
makeUniversal(y1)
makeUniversal(yn)


# when isMainModule:
  # echo angle(@[complex(12.0, 4.0), complex(7.0, 8.0)].toTensor)
  # echo abs(@[complex(12.0, 4.0), complex(7.0, 8.0)].toTensor)



