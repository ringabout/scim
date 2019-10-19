import streams, arraymancer


const
  WAVE_FORMAT_PCM = 0x0001
  WAVE_FORMAT_IEEE_FLOAT = 0x0003
  WAVE_FORMAT_EXTENSIBLE = 0xfffe
  KNOWN_WAVE_FORMATS = (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)


type
  WavFormatError* = Exception
  WavKind = enum
    WavUint8, WavInt16, WavInt32
  WavData*[T] = tuple
    data: Tensor[T]
    rate: uint32
  Wav* = ref object
    case kind*: WavKind
    of WavInt16:
      data16*: Tensor[int16]
    of WavInt32:
      data32*: Tensor[int32]
    of WavUint8:
      data8*: Tensor[uint8]
    rate*: uint32



proc readWav*(fileName: string): Wav =
  let strm = newFileStream(open(fileName))
  defer: strm.close()

  let
    chunkID = strm.readStr(4)
    chunkSize = strm.readUint32()
    format = strm.readStr(4)

    subchunk1ID = strm.readStr(4)
    subchunk1Size = strm.readUint32()
    audioFormat = strm.readUint16()
    numChannels = strm.readUint16()
    sampleRate = strm.readUint32()
    byteRate = strm.readUint32()
    blockAlign = strm.readUint16()
    bitsPerSample = strm.readUint16()

    subchunk2ID = strm.readStr(4)
    subchunk2Size = int strm.readUint32()

  # echo chunkID
  # echo chunkSize
  # echo format
  # echo subchunk1ID
  # echo subchunk1Size

  # echo audioFormat
  # echo numChannels
  # echo sampleRate
  # echo byteRate
  # echo blockAlign
  # echo bitsPerSample

  # echo subchunk2ID
  # echo subchunk2Size
  if audioFormat == WAVE_FORMAT_PCM:
    discard
  elif audioFormat == WAVE_FORMAT_IEEE_FLOAT:
    assert false, "please implement WAVE_FORMAT_IEEE_FLOAT"
    discard


  assert chunkID == "RIFF"
  assert format == "WAVE"
  assert subchunk1ID == "fmt "
  # assert audioFormat == 1
  assert subchunk2ID == "data"


  case bitsPerSample:
  of 16:
    var data: seq[int16]
    for _ in 1 .. (subchunk2Size div 2):
      data.add(strm.readInt16)
    result = Wav(kind: WavInt16, data16: data.toTensor, rate: sampleRate)
  of 32:
    var data: seq[int32]
    for _ in 1 .. (subchunk2Size div 4):
      data.add(strm.readInt32)
    result = Wav(kind: WavInt32, data32: data.toTensor, rate: sampleRate)
  of 8:
    var data: seq[uint8]
    for _ in 1 .. subchunk2Size:
      data.add(strm.readUint8)
    result = Wav(kind: WavUint8, data8: data.toTensor, rate: sampleRate)
  else:
    raise newException(WavFormatError, "don't support this wav format")


#[
    Notes
    -----
    This function cannot read wav files with 24-bit data.
    Common data types: [1]_
    =====================  ===========  ===========  =============
         WAV format            Min          Max       NumPy dtype
    =====================  ===========  ===========  =============
    32-bit floating-point  -1.0         +1.0         float32
    32-bit PCM             -2147483648  +2147483647  int32
    16-bit PCM             -32768       +32767       int16
    8-bit PCM              0            255          uint8
    =====================  ===========  ===========  =============
    Note that 8-bit PCM is unsigned.
]#

proc readWavData*[T](fileName: string): WavData[T] =
  let strm = newFileStream(open(fileName))
  defer: strm.close()

  let
    chunkID = strm.readStr(4)
    chunkSize = strm.readUint32()
    format = strm.readStr(4)

    subchunk1ID = strm.readStr(4)
    subchunk1Size = strm.readUint32()
    audioFormat = strm.readUint16()
    numChannels = strm.readUint16()
    sampleRate = strm.readUint32()
    byteRate = strm.readUint32()
    blockAlign = strm.readUint16()
    bitsPerSample = strm.readUint16()

    subchunk2ID = strm.readStr(4)
    subchunk2Size = int strm.readUint32()

  # echo chunkID
  # echo chunkSize
  # echo format
  # echo subchunk1ID
  # echo subchunk1Size

  # echo audioFormat
  # echo numChannels
  # echo sampleRate
  # echo byteRate
  # echo blockAlign
  # echo bitsPerSample

  # echo subchunk2ID
  # echo subchunk2Size
  if audioFormat == WAVE_FORMAT_PCM:
    discard
  elif audioFormat == WAVE_FORMAT_IEEE_FLOAT:
    assert false, "please implement WAVE_FORMAT_IEEE_FLOAT"
    discard


  assert chunkID == "RIFF"
  assert format == "WAVE"
  assert subchunk1ID == "fmt "
  # assert audioFormat == 1
  assert subchunk2ID == "data"

  var data: seq[T]
  case bitsPerSample:
    of 16:
      for _ in 1 .. (subchunk2Size div 2):
        data.add T(strm.readInt16)
    of 32:
      for _ in 1 .. (subchunk2Size div 4):
        data.add T(strm.readInt32)
    of 8:
      for _ in 1 .. subchunk2Size:
        data.add T(strm.readUint8)
    else:
      raise newException(WavFormatError, "don't support this wav format")
  result = (data.toTensor, sampleRate)



proc writeWav*[T](fileName: string, rate: uint32, data: Tensor[T]) =
  let strm = newFileStream(open(fileName, fmWrite))
  defer: strm.close()
  var
    channels: uint16
    subchunk2Size: uint32
  if data.rank == 1:
    channels = 1
  elif data.rank == 2:
    channels = 2
  subchunk2Size = uint32 (data.size * sizeof(T))


  let
    bitDepth = uint16 16
    bytesPerSecond = uint32 rate * (bitDepth div 8) * channels
    blockAlign = uint16 channels * (bitDepth div 8)

  # chunkID
  strm.write("RIFF")
  # chunkSize
  strm.write(36+subchunk2Size)
  # format
  strm.write("WAVE")
  # subchunk1ID
  strm.write("fmt ")
  # subchunk1Size 16 for PCM
  strm.write(uint32(16))
  # audioFormat 1 for PCM
  strm.write(uint16(1))
  # numChannels
  strm.write(channels)
  # sampleRate
  strm.write(rate)
  # byteRate
  strm.write(bytesPerSecond)
  # blockAlign
  strm.write(blockAlign)
  # bitsPerSample
  strm.write(uint16(sizeof(T) * 8))
  # subchunk2ID
  strm.write("data")
  # subchunk2Size
  strm.write(subchunk2Size)
  for chunk in data.items:
    strm.write(chunk)



when isMainModule:
  # let temp: Tensor[int16] = readWav("t1.wav")[1]
  # echo temp[1 .. 10]
  # import timeit
  # # echo timeGo(readWav("t1.wav"))

  var d1 = readWav("skip.wav")
  writeWav("write.wav", d1.rate, d1.data16)
  var d2 = readWav("write.wav")
  echo d1.data16 == d2.data16