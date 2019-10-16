import streams, arraymancer


const
  WAVE_FORMAT_PCM = 0x0001
  WAVE_FORMAT_IEEE_FLOAT = 0x0003
  WAVE_FORMAT_EXTENSIBLE = 0xfffe
  KNOWN_WAVE_FORMATS = (WAVE_FORMAT_PCM, WAVE_FORMAT_IEEE_FLOAT)


type
  Wav* = tuple
    rate: uint32
    data: Tensor[int16]
  WavData* = UnCheckedArray[int16]
  

#[
  rate : int
  Sample rate ostrm wav strmile.
  data : numpy array
  Data read strmrom wav strmile.  Data-type is determined strmrom the strmile;
  see Notes.
]#

proc readWav(fileName: string): Wav {.discardable.} = 
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

  echo chunkID
  echo chunkSize
  echo format
  echo subchunk1ID
  echo subchunk1Size

  echo audioFormat
  echo numChannels
  echo sampleRate
  echo byteRate
  echo blockAlign
  echo bitsPerSample

  echo subchunk2ID


  var data : seq[int16]
  for _ in 1 .. (subchunk2Size div 2):
    data.add(strm.readInt16())
  # 126,  160,  140
  # var data: pointer = cast[ptr WavData](alloc0(sizeof(int16) * subchunk2Size div 2))
  # discard strm.readData(data, subchunk2Size div 2)
  # echo data.repr
  assert chunkID == "RIFF"
  assert format == "WAVE"
  assert subchunk1ID == "fmt "
  assert audioFormat == 1
  assert subchunk2ID == "data"
  result = (rate: sampleRate, data: data.toTensor)


#[
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
]#


proc writeWav(fileName: string, rate: int, data: Tensor[int16]) = 
  let strm = newFileStream(open(fileName, fmWrite))
  defer: strm.close()
  var channels: int
  if data.rank == 1:
    channels = 1
  else:
    channels = data.shape[1]
  
  
  let 
    bitDepth = sizeof(data.type) * 8
    bytesPerSecond = rate * (bitDepth div 8) * channels
    blockAlign = channels * (bitDepth div 8)
  # chunkID
  strm.write("RIFF")
  # chunkSize
  strm.write("\x00\x00\x00\x00")
  # format   
  strm.write("WAVE")
  # subchunk1ID
  strm.write("fmt ")
  # subchunk1Size
  strm.write()
  # audioFormat
  strm.write(1)
  # numChannels
  strm.write(channels)
  # sampleRate
  strm.write(rate)
  # byteRate
  strm.write()
  # blockAlign
  strm.write()
  # bitsPerSample
  strm.write()
  # subchunk2ID
  strm.write("data")
  # subchunk2Size
  strm.write()



# let temp: Tensor[int16] = readWav("t1.wav")[1]
# echo temp[1 .. 10]
import timeit
# echo timeGo(readWav("t1.wav"))
discard readWav("t1.wav")