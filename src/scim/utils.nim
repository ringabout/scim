import arraymancer, math, fft, complex, sugar


const
  ## https://www.advanpix.com/2015/11/11
  COF = [
    1.00000000000000000000000000000000000e+00,
    2.50000000000000000000000000000000000e-01,
    2.77777777777777777777777777777777779e-02,
    1.73611111111111111111111111111111083e-03,
    6.94444444444444444444444444444449108e-05,
    1.92901234567901234567901234567853905e-06,
    3.93675988914084152179390274631572236e-08,
    6.15118732678256487780297303953307469e-10,
    7.59405842812662330592959640146193002e-12,
    7.59405842812662330592959486994771411e-14,
    6.27608134555919281481787961562157257e-16,
    4.35838982330499501028963901609513258e-18,
    2.57892888952958284633290216759752759e-20,
    1.31578004567835859498063743807352147e-22,
    5.84791131412603820814035808275571168e-25,
    2.28434035708048361004361404800451548e-27,
    7.90429189301205834423041758728181758e-30,
    2.43959626327530221779537803521876900e-32,
    6.75788438580533320595854894826703327e-35,
    1.68947109644654239602153081596915483e-37,
    3.83100021886675733871724163689480203e-40,
    7.91528970342866904594413272561413611e-43,
    1.49627405863018698139365798878928887e-45,
    2.59769774665516078740718557214698854e-48,
    4.15632134882805917464119034520347550e-51,
    6.14832848022015178960667680429552612e-54,
    8.43488878141065428604958554806851059e-57,
    1.07486432616861166276174172563459685e-59,
    1.28666328289305198822024236233461582e-62,
    1.37245729295774874628038022579554422e-65,
    1.71642649655915754818217188454585521e-68,
    6.42133551177881248636744515323368258e-72,
    2.93990539325117515154720726204799338e-74
    ]


  COF1 = [
    3.33094095073755755465308169515679754e-01,
    -4.87305575020532173309650536123692992e+00,
    1.16765891025166570372673307538545807e+00,
    2.64107028015824651013530450189957215e+03,
    3.29947497422711739169516563125826777e+02,
    1.85360637074230573336349486640079127e+02,
    1.92716946160432511352351625380961688e+02,
    2.94374157084426440123892033050844719e+02,
    5.94369974672125230816192833403509948e+02,
    1.49338188597066557028520330783876602e+03,
    4.49070212230053277828526595070904928e+03,
    1.57364639902573996182985842231707641e+04,
    6.09475654654342361774769290134171804e+04,
    6.74843325301108470040298999048635726e+05,
    -6.57083582192662169318674677284026011e+07,
    9.83172447953771800483159480192563859e+09,
    -1.23905750827828029147738416146512108e+12,
    1.35560883142284024977500631878243972e+14,
    -1.29343718391927120913858066893326175e+16,
    1.08130315540350608880516332890283373e+18,
    -7.95087530860069987411846664771880366e+19,
    5.15861021545066736428201155306431096e+21,
    -2.96090425993185982903689108288073850e+23,
    1.50651836855858836083071762402341231e+25,
    -6.80526037590684842951323289698851799e+26,
    2.73203132967909651102779478756247727e+28,
    -9.75309923194275182518137234542338189e+29,
    3.09640722254712498146426101828630546e+31,
    -8.73913695514222641997435352487537101e+32,
    2.19076945578257225990373580600588103e+34,
    -4.87120899281572328851130765475319009e+35,
    9.58810466892032483163229870526807875e+36,
    -1.66627420459640895942341335296132587e+38,
    2.54807715779107313134088724774425453e+39,
    -3.41409756541742494848545251263381364e+40,
    3.98660860345465130692055711924224814e+41,
    -4.02958459088507064758116007151695612e+42,
    3.49565245578103887077950778697885925e+43,
    -2.57418585602707614088803903097321313e+44,
    1.58622500396250248773886354120038387e+45,
    -8.02304482873909934532998282613480285e+45,
    3.24266546042741906420052016688464442e+46,
    -1.00665425805861133618962888863945150e+47,
    2.25310991683789163051367847410612888e+47,
    -3.23574657445200742953438267415045330e+47,
    2.23881303334352384470482998051927380e+47
    ]

  COF2 = [
    8.34943076824502833362836053862403964e-01,
    -1.23193072119209042422963284443456931e+01,
    4.40809330596253724239279903622218481e+00,
    6.62043547609998683454530303968680120e+03
    ]


type
  NotImplementError* = Exception
  WindowKind* = enum
    Kaiser, Rect, Hanning, Hamming, Blackman, BartLett
  EnergyKind* = enum
    AbsKind, SquareKind


proc honor(coeff: openArray[float64], x: float64): float64 =
  let n = coeff.len - 1
  result = coeff[n]
  for i in countdown(n - 1, 0, 1):
    result = coeff[i] + result * x



proc besselt0*(x: float64): float64 {.discardable.} =
  ## the modified zeroth-order Bessel function.
  var b: float64 = abs(x)
  if x < 15.5:
    b = (b / 2) ^ 2
    result = b * honor(COF, b) + 1.0
  else:
    b = 1 / b
    result = honor(COF1, b) / honor(COF2, b) * exp(x) * pow(x, -0.5)



proc kaiser*[T: SomeFloat](n: int, beta: float): Tensor[T] =
  if n == 1: return ones[T](1)
  result = newTensor[T](1, n)
  let alpha = (n - 1) / 2
  for i in 0 ..< n:
    result[0, i] = besselt0(beta * sqrt(1.0-((T(i)-alpha)/alpha) ^ 2)) /
        besselt0(beta)



proc hanning*[T: SomeFloat](n: int): Tensor[T] {.discardable.} =
  ## Creates a hanning window of length n.
  ## echo hanning[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](1, n)
  for i in 0 ..< n:
    result[0, i] = 0.5 - 0.5 * cos(2 * i / (n - 1) * Pi)


proc hamming*[T: SomeFloat](n: int): Tensor[T] {.discardable.} =
  ## Creates a hamming window of length n.
  ## echo hamming[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](1, n)
  for i in 0 ..< n:
    result[0, i] = 0.54 - 0.46 * cos(2 * i / (n - 1) * Pi)



proc rect*[T: SomeFloat](n: int): Tensor[T] =
  ## Creates a rect window of length n.
  ## echo rect[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](1, n)
  for i in 0 ..< n:
    result[0, i] = T(1)


proc bartlett*[T: SomeFloat](n: int): Tensor[T] =
  ## Creates a rect window of length n.
  ## echo rect[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](1, n)
  for i in 0 ..< n:
    result[0, i] = 1 - 2 / (n - 1) * abs((2 * i - n + 1) / 2)

proc blackman*[T: SomeFloat](n: int): Tensor[T] =
  ## Creates a rect window of length n.
  ## echo rect[float64](12)
  if n == 1: return ones[T](1)
  result = newTensor[T](1, n)
  for i in 0 ..< n:
    result[0, i] = 0.42 - 0.5 * cos(2 * i / (n - 1) * Pi) +
            0.08 * cos(4 * i / (n - 1) * Pi)


proc chooseWindow*[T: SomeFloat](n: int,
          kind: WindowKind, beta: float = 10): Tensor[T] =
  case kind
  of Kaiser:
    result = kaiser[T](n, beta)
  of Rect:
    result = rect[T](n)
  of Hanning:
    result = hanning[T](n)
  of Hamming:
    result = hamming[T](n)
  of Blackman:
    result = blackman[T](n)
  of Bartlett:
    result = bartlett[T](n)


proc preEmphasis*[T: SomeFloat](input: Tensor[T], factor: float): Tensor[T] =
  for i in 1 ..< input.size:
    input[0, i] -= input[0, i - 1] * factor.T



proc enFrame*[T: SomeFloat](input: Tensor[T], nFrameLength: int,
              nFrameInc: int, windowKind: WindowKind = Hamming): Tensor[T] =
  ## frames = length / nFrameLength
  var data = input
  if data.rank == 1:
    data = data.reshape(1, data.shape[0])
  elif data.rank >= 2:
    ## later implement mono
    raise newException(NotImplementError, "not implement the rank of input is more than 2")
  let w = chooseWindow[T](nFrameLength, windowKind)
  let length = data.shape[1]
  let frames = ((length - nFrameLength) div nFrameInc + 1) + 1
  let paddingLength = (frames - 1) * nFrameInc + nFrameLength - length
  result = newTensor[T](frames, nFrameLength)
  let paddingData = concat[T](data, zeros[T](1, paddingLength), axis = 1)
  for i in 0 ..< frames:
    let inc = i * nFrameInc
    result[i, _] = paddingdata[0, inc ..< inc + nFrameLength] .* w


proc squareSum*[T](t: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  t.reduce_axis_inline(axis):
    x += square(y)


proc absSum*[T](t: Tensor[T], axis: int): Tensor[T] {.noinit.} =
  t.abs.sum(axis) / t.shape[axis].T

proc normalize*[T](t: Tensor[T]): Tensor[T] {.noinit.} =
  t / t.max

proc getFrameEnergy*[T: SomeFloat](input: Tensor[T],
                    means: EnergyKind = AbsKind,
                        normalFlag: bool = true): Tensor[T] =
  case means
  of AbsKind:
    result = input.absSum(axis = 1)
  of SquareKind:
    result = input.squareSum(axis = 1)
  if normalFlag:
    result = result.normalize


proc zeroCrossing*[T: SomeFloat](input: Tensor[T]): Tensor[int] =
  assert input.rank == 2
  let
    rows = input.shape[0]
    cols = input.shape[1]
  result = newTensor[int](1, rows)
  for i in 0 ..< rows:
    for j in 1 ..< cols:
      result[0, i] += ord(input[i, j-1] * input[i, j] < 0)

proc zeroCrossingRate*[T: SomeFloat](input: Tensor[T]): Tensor[float] =
  assert input.rank == 2
  let
    rows = input.shape[0]
    cols = input.shape[1]
  result = newTensor[float](1, rows)
  for i in 0 ..< rows:
    for j in 1 ..< cols:
      result[0, i] += float(input[i, j-1] * input[i, j] < 0)
    result[0, i] /= cols.float

proc autoCorrlation*[T: SomeFloat](input: Tensor[T]): Tensor[float] =
  assert input.rank == 2
  let
    rows = input.shape[0] #rows
    cols = input.shape[1]
  result = newTensor[float](1, rows)
  for i in 0 ..< rows:
    for k in 0 ..< cols:
      for j in 0 ..< cols - k:
        result[0, i] += input[i, k + j] * input[i, j]


proc averMagnDiff*[T: SomeFloat](input: Tensor[T]): Tensor[float] =
  assert input.rank == 2
  let
    rows = input.shape[0] #rows
    cols = input.shape[1]
  result = newTensor[float](1, rows)
  for i in 0 ..< rows:
    for k in 0 ..< cols:
      for j in 0 ..< cols - k:
        result[0, i] += abs(input[i, k + j] - input[i, j])



proc frame2Time*(frames, nFrameLength, nFrameInc, rate: int): seq[float] =
  for i in 0 ..< frames:
    result.add (i * nFrameInc + nframeLength div 2) / rate


proc stftms*[T: SomeFloat](input: Tensor[T]): Tensor[Complex[T]] =
  assert input.rank == 2
  # TODO fft
  let
    rows = input.shape[0]
    cols = input.shape[1]
  var length = 2 ^ int(log2(float(cols)))
  if length < cols:
    length *= 2
  result = newTensor[Complex[T]](rows, length)
  for i in 0 ..< rows:
    for j in 0 ..< length:
      result[i, j] = input[i, _].fft[0, j]

proc periodogram*[T](input: Tensor[Complex[T]]): Tensor[T] =
  assert input.rank = 2
  let 
    rows = input.shape[0]
    cols = input.shape[1]
  result = newTensor[T](1, rows)
  for i in 0 ..< rows:
    for j in 0 .. < cols:
      result[0, i] += abs(input[i, j]) ^ 2
  # ?
  result.map(x=>x/cols) 

proc frameCepstrum*[T](input: Tensor[Complex[T]]): Tensor[T] = 
  assert input.rank = 2
  let 
    rows = input.shape[0]
    cols = input.shape[1]
  result = newTensor[T](rows, cols)   
  for i in 0 ..< rows:
    for j in 0 .. < cols:
      result[i, j] = log2(abs(input[i, j]))
  result.ifft.map(x=>x.re)


when isMainModule:
  # import timeit
  # var m = monit("utils")
  # m.start()
  # hanning[float](12000)
  # m.finish()
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

  discard kaiser[float](12, 14)

  # echo "test1"
  # echo timeGo(hanning[float](12000))
  # echo "test2"
  # echo timeGo(hamming[float](12000))
