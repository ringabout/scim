import math, complex, timeit, sugar, arraymancer, sequtils
 

  
type
  ComplexType = Complex[float64] | Complex[float32]




proc bitReverseCopy[T: ComplexType](x: seq[T]): seq[T] = 
  let n = x.len
  var mid = x
  var
    k: int
    temp: T
    j: int = 0
  for i in 0 ..< n - 1:
    if i < j:
      temp = mid[j]
      mid[j] = mid[i]
      mid[i] = temp
    k = n shr 1
    while j >= k:
      j -= k
      k = k shr 1
    j += k
  for item in mid:
    result.add item



proc fftAid[T: ComplexType](x: seq[T], flag: float = -1): Tensor[T] =
  var 
    n = x.len
    n1 = int(log2(n.float32))
    n2 = 2 ^ n1
    padding_length: int
    temp = x
  if n != n2:
    padding_length = 2 * n2 - n
    n = 2 * n2
    n1 += 1
  for i in 1 .. padding_length:
    temp.add(complex(0.0))
  temp = bitReverseCopy[T](temp)
  for s in 1 .. n1:
    let 
      m = 2 ^ s
      # flag * 2 * Pi 
      wm = exp(complex(0.0, flag * 2.0 * Pi / float(m)))
 
    for k in countup(0, n - 1 , m):
      var w = complex(1.0)
      let m2 = m div 2
      for j in 0 ..< m2:
        let 
          t = w * temp[k + j + m2]
          u = temp[k + j]
        temp[k + j] = u + t
        temp[k + j + m2] = u - t
        w = w * wm
  result = temp.toTensor.reshape(1, temp.len)


proc fft*[T:ComplexType](x: seq[T] | Tensor[T]): Tensor[Complex[float64]] =
  var temp: seq[T]
  when x is seq:
    temp = x
  elif x is Tensor:
    temp = x.toRawSeq
  result = temp.fftAid(-1)

proc fft*[T: SomeFloat](x: seq[T] | Tensor[T]): Tensor[Complex[float64]] =
  result = fft(x.map(t=>t.complex))

proc ifft*[T:ComplexType](x: seq[T] | Tensor[T]): Tensor[Complex[float64]] =
  var temp: seq[T]
  when x is seq:
    temp = x
  elif x is Tensor:
    temp = x.toRawSeq
  # when T is SomeFloat:
  #   temp = temp.map(x=>complex(x))
  result = temp.fftAid(1).map(item => item / temp.len.float64)

proc ifft*[T: SomeFloat](x: seq[T] | Tensor[T]): Tensor[Complex[float64]] =
  result = ifft(x.map(t=>t.complex))

# echo a

when isMainModule:
  var a = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0].map(x=>complex(x))
  var b = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0].map(x=>complex(x))
  var c = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0].map(x=>complex(x))
  echo ifft(fft(a))
  echo ifft(fft(b))
  echo ifft(fft(c))


# echo timeGo(fft(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
# echo timeGo(fft1(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))



# echo timeGo(fft(aaa))
# echo timeGo(fft1(aaa))
# var res = fft1(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

# for i in ifft(res):
#   echo formatFloat(abs(i), ffDecimal, 10)