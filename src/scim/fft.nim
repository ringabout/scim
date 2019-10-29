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

proc rfft*[T: SomeFloat](input: Tensor[T]): Tensor[Complex[float64]] =
  assert input.rank == 2
  var
    n = input.shape[1]
    half = n div 2
    A = newTensor[Complex[T]](half)
    B = newTensor[Complex[T]](half)
    IA = newTensor[Complex[T]](half)
    IB = newTensor[Complex[T]](half)
    X = newTensor[Complex[T]](1, half)
  result = newTensor[Complex[T]](1, n)
  for k in 0 ..< half:
    let 
      coeff = 2.0 * float(k) * PI / float(n)
      cosPart = 0.5 * cos(coeff)
      sinPart = 0.5 * sin(coeff)
    A[k] = complex(0.5 - sinPart, -cosPart)
    B[k] = complex(0.5 + sinPart, cosPart)
    IA[k] = conjugate(A[k])
    IB[k] = conjugate(B[k])
  for i in 0 ..< half:
    X[0, i] = complex(input[0, 2 * i], input[0, 2 * i + 1])
  var temp = newTensor[Complex[T]](1, half + 1)
  # TODO not 2 ^ n
  temp[0, 0 ..< half] = X.fft
  temp[0, half] = temp[0, 0]
  result[0, 0] = temp[0, 0] * A[0] + conjugate(temp[0, half]) * B[0]
  for j in 1 ..< half:
    result[0, j] = temp[0, j] * A[j] + conjugate(temp[0, half - j]) * B[j]
    result[0, n-j] = conjugate(result[0, j])
  result[0, half] = complex(temp[0, 0].re - temp[0, 0].im, 0.0)
  

    
 


when isMainModule:
  # var a = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0].map(x=>complex(x))
  # var b = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0].map(x=>complex(x))
  var c = @[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0].toTensor.reshape(1, 8)
  # echo ifft(fft(a))
  # echo ifft(fft(b))
  echo fft(c)
  echo rfft(c)
  # echo ifft(rfft(c))


# echo timeGo(fft(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
# echo timeGo(fft1(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))



# echo timeGo(fft(aaa))
# echo timeGo(fft1(aaa))
# var res = fft1(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

# for i in ifft(res):
#   echo formatFloat(abs(i), ffDecimal, 10)