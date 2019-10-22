import arraymancer, math, complex

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



makeUniversal(j0)
makeUniversal(j1)
makeUniversal(jn)
makeUniversal(y0)
makeUniversal(y1)
makeUniversal(yn)


when isMainModule:
  echo angle(@[complex(12.0, 4.0), complex(7.0, 8.0)].toTensor)
  echo abs(@[complex(12.0, 4.0), complex(7.0, 8.0)].toTensor)



