from _common_data_preprocessing import *
import pickle


""" 
globals then atoms then labels
"""


#This isn't writing the predictions yet. I'm going to dump those in a separate file for ease of thought
def format(rFile, wFile, sym, topo):
    global globals
    global maxSize
    global atomSize
    with open(rFile, 'r', newline='') as file:
        reader = csv.reader(file)
        outPutls = []
        for row in reader:
            poscar = list(map(lambda a: a.strip(), row[0].split("\\n")))

            #Not the most efficient encoding, but is fine
            arr = [0]*((globals["dataSize"] + atomSize)*maxSize + globals["labelSize"])#9 init vals, maxSize spots for atoms, 5 final categories - 1 for the simple topo

            glData = getGlobalData(poscar, row, sym)

            atoms = poscar[5].split()
            numbs = poscar[6].split()

            total = 0
            for i in range(len(numbs)):
                total+=int(numbs[i])
                numbs[i] = total
            

            curIndx = 0
            atomType = 0
            for i in range(total):
                curIndx+=1
                if curIndx > numbs[atomType]:
                    atomType+=1

                arr[i*(globals["dataSize"] + atomSize):(i+1)*(globals["dataSize"] + atomSize)] = serializeAtom(atoms[atomType], poscar, i) + glData
            
            #This should be dumping an array into the end of the array
            arr[-globals["labelSize"]:] = convertTopoToIndex(row, topo)
            outPutls.append(arr)
        
        with open(wFile, 'wb') as wFile:
            pickle.dump(outPutls, wFile)

cmd_line(format, "csnn")




from typing import Iterator, Mapping, Tuple
from _common_ml import *
import csv
from absl import app
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


atomSize = 27

# we have l | l l l so the sizes are the same
def skipit(width, layers):
  first = hk.Sequential([hk.Linear(width), jax.nn.relu])

  rest = hk.Sequential([
    *[a for _ in range(layers - 1) for a in (hk.Linear(width), jax.nn.leaky_relu)]
  ])
  return lambda a : first(a) + rest(a)


def MAB(X, Y, is_self_attention=False):
  if is_self_attention:
    Y = X

  attention_output = hk.MultiHeadAttention(num_heads=8, key_size=128, w_init_scale=1.0, model_size=X.shape[-1])(X, Y, Y)
  H = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(X + hk.Linear(X.shape[-1])(attention_output))
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(H + hk.nets.MLP([256, 128, 128, H.shape[-1]], activation=jax.nn.leaky_relu)(H))


def SAB(X):
  return MAB(X, X, is_self_attention=True)

def ISAB(X, num_inducing_points=16):
  inducing_points = hk.get_parameter("inducing_points", shape=[num_inducing_points, X.shape[-1]], init=hk.initializers.RandomNormal(stddev=0.1))
  H = MAB(inducing_points, X)
  return MAB(X, H, is_self_attention=False)

def PMA(Z):
  num_seeds=1 #apply SAB after if you ever need more vectors out the end
  seed_vectors = hk.get_parameter("seed_vectors", shape=[num_seeds, Z.shape[-1]], init=hk.initializers.RandomNormal())
  return MAB(seed_vectors, hk.Linear(Z.shape[-1])(Z), is_self_attention=False)

def encoder(X):
  X = ISAB(ISAB(ISAB(X)))
  return X

def decoder(Z):
  #Z = hk.nets.MLP([512, 256, 64], activation=jax.nn.leaky_relu)(Z)#check rep dim - prob not needed
  Z = PMA(Z)#SAB(PMA(Z))
  Z = hk.nets.MLP([512, 256, 64], activation=jax.nn.leaky_relu)(Z)
  return hk.Linear(globals["labelSize"])(Z)#implicit global var to avoid passing static_args

@jax.vmap
def set_transformer_single(X):
  # Encoder
  Z = encoder(X)

  # Decoder
  output = decoder(Z)

  return output

def set_transformer(batch, output_size):
  # Reshape input to (batch_size, n, d_x)
  reshaped_batch = jnp.reshape(batch, (batch.shape[0], 60, globals["dataSize"] + atomSize))

  # Apply set_transformer_single to each example in the batch using jax.vmap
  output = set_transformer_single(reshaped_batch)

  return output


def net_fn(batch: jnp.ndarray) -> jnp.ndarray:
  return set_transformer(batch, globals["labelSize"])


#maybe a better way to shuffle data exists, but... ?
#Also, could try vmapping
def mixAtoms(listOfData):
  return listOfData

def main(obj):
  # Make the network and optimiser.
  print("entered function")
  net = hk.transform(net_fn, apply_rng=True)
  opt = optax.adam(1e-3)

  # Training loss (cross-entropy).
  def loss(params: hk.Params, batch: np.ndarray, labels: np.ndarray) -> jnp.ndarray:
    global globals
    """Compute the loss of the network, including L2."""
    logits = net.apply(params, rng, batch)
    #The labels are prehotted

    l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
    loss = jnp.mean(optax.softmax_cross_entropy(logits, labels))

    return loss + 1e-4 * l2_loss

  # Evaluation metric (classification accuracy).
  @jax.jit
  def accuracy(params: hk.Params, batch: np.ndarray, labels: np.ndarray) -> jnp.ndarray:
    global globals
    predictions = net.apply(params, rng, batch)
    hot_pred = jax.lax.map(
      lambda a: jax.lax.eq(jnp.arange(globals["labelSize"]), jnp.argmax(a)).astype(float),
      predictions)
    return jnp.mean(jnp.equal(hot_pred, labels).all(axis=1))

  @jax.jit
  def update(
      params: hk.Params,
      opt_state: optax.OptState,
      batch: np.ndarray,
      labels: np.ndarray,
  ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads = jax.grad(loss)(params, batch, labels)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state

  # We maintain avg_params, the exponential moving average of the "live" params.
  # avg_params is used only for evaluation (cf. https://doi.org/10.1137/0330046)
  @jax.jit
  def ema_update(params, avg_params):
    return optax.incremental_update(params, avg_params, step_size=0.001)

  print("getting datasets")
  # Make datasets.
  iterData = list(np.array_split(np.array(obj), 100))

  print("done splitting")

  # Initialize network and optimiser; note we draw an input to get shapes.
  rng = jax.random.PRNGKey(42)
  params = avg_params = net.init(rng, iterData[0][:,:-globals["labelSize"]])
  opt_state = opt.init(params)


  print("training")
  ii = 1
  # Train/eval loop.
  for step in range(100000):
    ii+=1
    if ii == len(iterData):
      ii = 1

    
    if step % 100 == 10:
      # Periodically evaluate classification accuracy on train & test sets.
      train_accuracy = accuracy(avg_params, 
                                mixAtoms(iterData[ii][:,:-globals["labelSize"]]),
                                          iterData[ii][:, -globals["labelSize"]:])
      test_accuracy = accuracy(avg_params,
                                mixAtoms(iterData[0][:,:-globals["labelSize"]]),
                                         iterData[0][:, -globals["labelSize"]:])
      train_accuracy, test_accuracy = jax.device_get(
          (train_accuracy, test_accuracy))
      print(f"[Step {step}] Train / Test accuracy: "
            f"{train_accuracy:.3f} / {test_accuracy:.3f}.")

    
    # Do SGD on a batch of training examples.
    params, opt_state = update(params, opt_state, 
                               mixAtoms(iterData[ii][:,:-globals["labelSize"]]), 
                                         iterData[ii][:, -globals["labelSize"]:])
    avg_params = ema_update(params, avg_params)



cmd_line(main, "csnn")
