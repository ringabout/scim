import math, complex, timeit, sequtils, sugar, strutils
 
# Works with floats and complex numbers as input
proc fft1[T: float | Complex64](x: openarray[T]): seq[Complex64] {.discardable.} =
  # 获取长度
  let n = x.len
  if n == 0: return
 
  result.newSeq(n)
 
  if n == 1:
    result[0] = (when T is float: complex(x[0]) else: x[0])
    return
 
  var evens, odds = newSeq[T]()
  for i, v in x:
    if (i and 1) == 0: evens.add v
    else: odds.add v
  var 
    (even, odd) = (fft1(evens), fft1(odds))
    w = complex(1.0)
    wm = exp(complex(0.0, -2 * Pi / float(n)))
  let halfn = n shr 1
 
  for k in 0 ..< halfn:
    let a = w * odd[k]
    result[k] = even[k] + a
    result[k + halfn] = even[k] - a
    w *= wm



proc ifftAid(x: seq[Complex64]): seq[Complex64] =
  let n = x.len
  if n == 0: return

  result.newSeq(n)

  if n == 1: 
    result[0] = x[0]
    return

  var evens, odds = newSeq[Complex64]()
  for i, v in x:
    if (i and 1) == 0: evens.add v
    else: odds.add v
  var
    (even, odd) = (ifftAid(evens), ifftAid(odds))
    w = complex(1.0)
    wm = exp(complex(0.0, 2 * Pi / float(n)))
  let half = n shr 1
  for k in 0 ..< half:
    let a = w * odd[k]
    result[k] = (even[k] + a) 
    result[k + half] = (even[k] - a) 
    w *= wm

proc ifft(x: seq[Complex64]): seq[Complex64] =
  return x.ifftAid.map(item => item / x.len.float)
  
  



type
  ComplexType = float | Complex64


proc bitReverseCopy[T: ComplexType](x: seq[T]): seq[Complex64] = 
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
    result.add complex(item)




proc fft[T: ComplexType](x: seq[T]): seq[Complex64] {.discardable.} =
  let n = x.len
  var a = bitReverseCopy(x)
  for s in 1 .. int(log2(n.float32)):
    let 
      m = 2 ^ s
      wm = exp(complex(0.0, -2 * Pi / float(m)))
    for k in countup(0, n - 1 , m):
      var w = complex(1.0)
      let m2 = m div 2
      for j in 0 ..< m2:
        let 
          t = w * a[k + j + m2]
          u = a[k + j]
        a[k + j] = u + t
        a[k + j + m2] = u - t
        w = w * wm
  return a

import arraymancer


# echo timeGo(fft(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))
# echo timeGo(fft1(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]))


var aaa: seq[float] = @[]
for i in 1 .. 2048:
  aaa.add(float(i))

# echo timeGo(fft(aaa))
# echo timeGo(fft1(aaa))
# var res = fft1(@[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

# for i in ifft(res):
#   echo formatFloat(abs(i), ffDecimal, 10)