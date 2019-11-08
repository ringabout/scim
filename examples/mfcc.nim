import arraymancer, sugar
import ../src/scim/utils, ../src/scim/scimio




import timeit
timeOnce("wave"):
  var 
    (data, rate) = readWavData[float]("test.wav")
    dmax = data.max
  data = data.map(x=>x/dmax)
  var input = enframe[float](data, 256, 80)
  let 
    fData = stftms(input)
    banks = 16
    res = frameMelCoeff[float](fData, cast[int](rate), banks)
echo res[0, 0]