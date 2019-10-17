# This is just an example to get you started. You may wish to put all of your
# tests into a single file, or separate them into multiple `test1`, `test2`
# etc. files (better names are recommended, just make sure the name starts with
# the letter 't').
#
# To run these tests, simply execute `nimble test`.

import unittest, math, sequtils, timeit, arraymancer




import scim
test "can add":
  check hanning[float](12000) == hamming[float](12000)

# test "can add":
#   check hanning[float](1200) == hamming[float](1200)
