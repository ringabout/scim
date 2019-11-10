import arraymancer, strformat


let
  d1 = "data.npy"
  d2 = "target.npy"
  ctx = newContext Tensor[float32]
  x = ctx.variable(read_npy[float32](d1).reshape(11634, 220))
var
  tmp = read_npy[int](d2).reshape(11634)
  y = newTensor[float32](tmp.shape[0], 6)

for i in 0 ..< tmp.shape[0]:
  y[i, tmp[i]] = 1'f32


# Create the autograd context that will hold the computational graph
network ctx, SpeechNet:
  layers:
    fc1: Linear(220, 640)
    fc2: Linear(640, 640)
    fc3: Linear(640, 64)
    fc4: Linear(64, 6)
  forward x:
    x.fc1.relu.fc2.relu.fc3.relu.fc4

let
  model = ctx.init(SpeechNet)
  optim = model.optimizerSGD(learning_rate = 1e-4'f32)

# ##################################################################
# Training
var y_pred_value: Tensor[int]
for t in 0 ..< 100:
  let
    y_pred = model.forward(x)
    loss = sigmoid_cross_entropy(y_pred, y)
  y_pred_value = y_pred.value.softmax.argmax(axis = 1).squeeze
  let
    score = accuracy_score[int](tmp, y_pred_value)
  echo &"Epoch {t}: score {score}"
  echo &"Epoch {t}: loss {loss.value[0]}"
  loss.backprop()
  optim.update()
echo y_pred_value