import streams
import arraymancer


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


proc writeWav(fileName: string, rate: int, data: seq[int16]) = 
  discard


# let temp: Tensor[int16] = readWav("t1.wav")[1]
# echo temp[1 .. 10]
import timeit
echo timeGo(readWav("t1.wav"))
