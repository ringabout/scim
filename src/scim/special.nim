import arraymancer, math, complex

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
proc roll*[T](t: Tensor[T], shift: int=1, axis:int= -1): Tensor[T] = 
  if shift == 0:
    return t 
  result = t
  if t.rank == 1:
    result = t.reshape(1, t.size)
  let 
    rows = result.shape[0]
    cols = result.shape[1]
  var s: int = shift
  while s < 0:
    s += cols
  s = s mod cols
  let n = gcd(cols, s)
  for i in 0 ..< rows:
    for j in 0 ..< n:
      let temp = result[i, j]
      var k: int = j
      while true:
        var t = k + s
        if t >= cols:
          t -= cols
        if t == j:
          break
        result[i, k] = result[i, t]
        k = t
      result[i, k] = temp




makeUniversal(j0)
makeUniversal(j1)
makeUniversal(jn)
makeUniversal(y0)
makeUniversal(y1)
makeUniversal(yn)


# when isMainModule:
  # echo angle(@[complex(12.0, 4.0), complex(7.0, 8.0)].toTensor)
  # echo abs(@[complex(12.0, 4.0), complex(7.0, 8.0)].toTensor)



