import neo

let
  m1 = randomMatrix(6, 9)
  m2 = randomMatrix(9, 6)
  v1 = randomVector(9)
echo m1.t # transpose, done in constant time without copying
echo m1 * m2
# echo m1 + m2.t
let m3 = m1.reshape(9, 6)
let m4 = v1.asMatrix(3, 3)
let v2 = m2.asVector
# import arraymancer

# let j = [0, 10, 20, 30].toTensor.reshape(4,1)
# let k = [0, 1, 2].toTensor.reshape(1,3)

# echo j * k