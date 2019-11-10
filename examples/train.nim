import arraymancer, strformat


let
  d1 = "data.npy"
  d2 = "target.npy"
  ctx = newContext Tensor[float32]
  x = ctx.variable(read_npy[float32](d1).reshape(11634, 220))
  y = read_npy[float32](d2).reshape(11634)


# Create the autograd context that will hold the computational graph
network ctx, TwoLayersNet:
  layers:
    fc1: Linear(11634, 200)
    fc2: Linear(200, 6)
  forward x:
    x.fc1.relu.fc2

let
  model = ctx.init(TwoLayersNet)
  optim = model.optimizerSGD(learning_rate = 1e-4'f32)

# ##################################################################
# Training

for t in 0 ..< 500:
  let
    y_pred = model.forward(x)
    loss = mse_loss(y_pred, y)

  echo &"Epoch {t}: loss {loss.value[0]}"

  loss.backprop()
  optim.update()