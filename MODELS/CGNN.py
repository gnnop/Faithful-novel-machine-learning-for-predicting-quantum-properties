import sys, os, shutil

if not __name__ == '__main__':
  os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

#standard thing that makes sure all dependencies resolve.
def copy_file_to_directories(filename, target_dirs):
    source_path = os.path.abspath(filename)
    print(f"Source path: {source_path}")
    if not os.path.exists(source_path):
        print(f"Source file {filename} does not exist.")
        return
    for dir in target_dirs:
        dir = os.path.join(os.path.abspath("."), dir)
        print(f"Target directory: {dir}")
        os.makedirs(dir, exist_ok=True)
        target_path = os.path.join(dir, os.path.basename(filename))
        shutil.copy(source_path, target_path)
        print(f"Copied {source_path} to {target_path}")

# Define the relative paths of the target directories
target_directories = ['.', 'input', 'output']
if __name__ == '__main__':
    copy_file_to_directories('../includes.py', target_directories)

#Now the rest of the file.

from includes import *
load_submodules() #this is a function in includes.py that loads all the submodules.


# hyperparameters
hp = {
    "dropoutRate": 0.1,
    "max_atoms": 40,
    "batch_size": 8,#This is now baked into the preprocessing
    "num_proc": 12,#This is the number of processes used in preprocessing
}

def GraphConvolution(update_node_fn: Callable,
                     aggregate_nodes_fn: Callable = jax.ops.segment_sum,
                     add_self_edges: bool = False,
                     symmetric_normalization: bool = True) -> Callable:
  """Returns a method that applies a Graph Convolution layer.

  Graph Convolutional layer as in https://arxiv.org/abs/1609.02907,
  NOTE: This implementation does not add an activation after aggregation.
  If you are stacking layers, you may want to add an activation between
  each layer.
  Args:
    update_node_fn: function used to update the nodes. In the paper a single
      layer MLP is used.
    aggregate_nodes_fn: function used to aggregates the sender nodes.
    add_self_edges: whether to add self edges to nodes in the graph as in the
      paper definition of GCN. Defaults to False.
    symmetric_normalization: whether to use symmetric normalization. Defaults to
      True.

  Returns:
    A method that applies a Graph Convolution layer.
  """

  def _ApplyGCN(graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    """Applies a Graph Convolution layer."""
    nodes, _, receivers, senders, _, _, _ = graph

    # First pass nodes through the node updater.
    nodes = update_node_fn(nodes)
    # Equivalent to jnp.sum(n_node), but jittable
    total_num_nodes = tree.tree_leaves(nodes)[0].shape[0]
    #In the original example, self-edges were included. 
    #based on how I arranged the data, it shouldn't be necessary.

    conv_senders = senders
    conv_receivers = receivers

    # pylint: disable=g-long-lambda
    if symmetric_normalization:
      # Calculate the normalization values.
      count_edges = lambda x: jax.ops.segment_sum(
          jnp.ones_like(conv_senders), x, total_num_nodes)
      sender_degree = count_edges(conv_senders)
      receiver_degree = count_edges(conv_receivers)

      # Pre normalize by sqrt sender degree.
      # Avoid dividing by 0 by taking maximum of (degree, 1).
      nodes = tree.tree_map(
          lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
          nodes,
      )
      # Aggregate the pre-normalized nodes.
      nodes = tree.tree_map(
          lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                       total_num_nodes), nodes)
      # Post normalize by sqrt receiver degree.
      # Avoid dividing by 0 by taking maximum of (degree, 1).
      nodes = tree.tree_map(
          lambda x:
          (x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]),
          nodes,
      )
    else:
      nodes = tree.tree_map(
          lambda x: aggregate_nodes_fn(x[conv_senders], conv_receivers,
                                       total_num_nodes), nodes)
    # pylint: enable=g-long-lambda
    return graph._replace(nodes=nodes)

  return _ApplyGCN

@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Edge update function for graph net."""
  net = hk.Sequential(
      [
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(128)])
  return net(feats)

@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Node update function for graph net."""
  net = hk.Sequential(
      [
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(128)])
  return net(feats)
# end edge_update_fn

@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
  """Global update function for graph net."""
  # MUTAG is a binary classification task, so output pos neg logits.
  net = hk.Sequential(
      [
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(128)])
  return net(feats)
# end update_global_fn

def net_fn(graph, is_training=False, dropout_rate=0):
  global globals

  #The shape of the graph is wrong.
  #The graph globals structure relies on the globals being in
  #a matrix. Uhhhh

  collector = jraph.GraphMapFeatures(
    hk.Sequential(
      [
      hk.Linear(128)]),
    hk.Sequential(
      [
      hk.Linear(128)]),
    hk.Sequential(
      [
      hk.Linear(512), jax.nn.leaky_relu,
      hk.Linear(512), jax.nn.leaky_relu,
      hk.Linear(512), jax.nn.leaky_relu,
      hk.Linear(256), jax.nn.leaky_relu,
      hk.Linear(128), jax.nn.leaky_relu,
      hk.Linear(globals["labelSize"])]))

  embedder = jraph.GraphMapFeatures(
      hk.Sequential(
      [
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512)]), 
      hk.Sequential(
      [
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512)]), 
      hk.Sequential(
      [
       hk.Linear(256), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512), jax.nn.leaky_relu,
       hk.Linear(512)]))
  net = jraph.GraphNetwork(
      update_node_fn=node_update_fn,
      update_edge_fn=edge_update_fn,
      update_global_fn=update_global_fn)
  
  x1 = embedder(graph)
  x2 = net(x1)
  x3 = collector(x2)

  #return graph
  return x3.globals * jraph.get_graph_padding_mask(x3)#Need to incorporate output dim somehow.

output_dim = 0#initialize before calling the network

def reflectAtomDist(lis):
    return map(lambda a: [a[0], -a[1], -a[2], -a[3]], lis)


def atomDistance(params, loc1, loc2):
    global completeTernary
    maxBondDistance = 3.1 #this seems to be reasonable, but the min problem is what to do with multiple bonding.
    #I found a paper where they just ignore the problem, so that's the first approach here;

    dist = []
    
    for i in completeTernary:
        dir = np.array(loc1) - np.array(loc2) + i[0]*np.array(params[0]) + i[1]*np.array(params[1]) + i[2]*np.array(params[2])
        if np.linalg.norm(dir) < maxBondDistance:
            dist.append([np.linalg.norm(dir),*dir])
    
    #Do I need the subtraction info?
    return dist

def _nearest_bigger_power_of_two(x: int) -> int:
  """Computes the nearest power of two greater than x for padding."""
  y = 2
  while y < x:
    y *= 2
  return y

def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple) -> jraph.GraphsTuple:
  """Pads a batched `GraphsTuple` to the nearest power of two"""
  # Add 1 since we need at least one padding node for pad_with_graphs.
  pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
  pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
  # Add 1 since we need at least one padding graph for pad_with_graphs.
  # We do not pad to nearest power of two because the batch size is fixed.
  pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
  return jraph.pad_with_graphs(graphs_tuple, pad_nodes_to, pad_edges_to,
                               pad_graphs_to)

def extractGraph(atomic_encoding, absolute_position, glob, label_size, axes):
    globalData = [0.0]*(len(glob) + label_size)
    globalData[0:len(glob)] = glob

    nodesArray = atomic_encoding

    
    #Now we do an n^2 operation on the atoms:

    senderArray = []
    receiverArray = []
    edgeFeatures = []

    for i in range(len(absolute_position)):
        for j in range(i + 1):
            ls = atomDistance(axes, absolute_position[i], absolute_position[j])
            #This is a list of
            senderArray.extend([i]*len(ls))
            receiverArray.extend([j]*len(ls))
            edgeFeatures.extend(ls)
            if i != j:
                senderArray.extend([j]*len(ls))
                receiverArray.extend([j]*len(ls))
                edgeFeatures.extend(reflectAtomDist(ls))

    #everything should be lined up now, but we need to add it to a graph

    return jraph.GraphsTuple(
        nodes=jnp.array(nodesArray), 
        senders=jnp.array(senderArray), 
        receivers=jnp.array(receiverArray), 
        edges=jnp.array(edgeFeatures), 
        globals=jnp.array([globalData]),
        n_node=jnp.array([len(nodesArray)]),
        n_edge=jnp.array([len(senderArray)]))

def getGlobalDataVector(poscar):
    return np.array([unpackLine(poscar[2]), unpackLine(poscar[3]), unpackLine(poscar[4])])

def processGraphBatch(graph):
  poscar = preprocessPoscar(graph)
  numbs = poscar[6].split()

  total = 0
  for ii in range(len(numbs)):
    total+=int(numbs[ii])
    numbs[ii] = total

  #fixed dimensional
  axes_ = getGlobalDataVector(poscar)
  pi_ = flatten([poscar_global.info(poscar) for poscar_global in poscar_globals] + [global_input.info(i) for global_input in global_inputs])
  inputs_ = np.array([sum(items, []) for items in zip(*[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics])])
  positions_ = [np.matmul(unpackLine(poscar[8+ii]), axes_) for ii in range(total)]

  #fixed dimensional
  go_ = []
  for func in range(len(global_outputs)):
    go_.append(global_outputs[func].info(graph))
  go_ = sum(go_, [])

  graph_ = extractGraph(inputs_, positions_, pi_, len(go_), axes_)

  return graph_, go_



if __name__ == '__main__':
  if not exists("CGNN.pickle"):

      #the atomic embeddings are assumed to be both input and available for all
      #poscar files. The global embeddings are assumed to not exist for all elements.
      #The poscar_globals is just the cell size. The others should be obvious.

      #Print the number of poscars total:
      setOfAllPoscars = getSetOfPoscars()
      print("There are", len(setOfAllPoscars), "poscars total.")

      ids = set.intersection(*flatten([i.valid_ids() for i in global_inputs] + [i.valid_ids() for i in global_outputs]))

      print("There are", len(ids), "poscars with all inputs and outputs.")

      #There are how many poscars missing:
      ids = set.intersection(ids, setOfAllPoscars)
      print("There are", len(ids), "poscars with all inputs and outputs and all poscars.")


      data = []
      labels = []
      go_types = [i.classifier() for i in global_outputs]
      go_nums = [len(i.info(str(1))) for i in global_outputs]

      def accumulate_data(args):
        global data
        global labels

        #if len(data) % 100 == 0:
        print(len(data), "batches processed")
          
        data.append(args[0])
        labels.append(args[1])

      pool = Pool(processes=hp["num_proc"])
      #Was going to batch the graphs, but padding them is incredibly memory intensive

      for index in list(ids):
          pool.apply_async( processGraphBatch, args=(index,) , callback=accumulate_data )

      pool.close()
      pool.join()

      with open("CGNN.pickle", "wb") as f:
          pickle.dump({
              "data": data,
              "labels": labels,
              "labels_types": go_types,
              "labels_nums": go_nums
          },f)


  with open("CGNN.pickle", "rb") as f:
      data = pickle.load(f)
      inputs = data["data"]
      labels = data["labels"]
      label_types = data["labels_types"]
      label_nums = data["labels_nums"]

      output_dim = sum(label_nums)


      # Split the data into training and validation sets
      ##X_train, y_train, add_train, X_val, y_val, add_val = partition_dataset(0.4, data, labels, additional_data)
      X_train, y_train, X_val, y_val = partition_dataset(0.1, inputs, labels)

      #Now, attempt to load the model CGNN.params if it exists, otherwise init with haiku
      # Initialize the network
      net = hk.transform(net_fn)
      rng = jax.random.PRNGKey(0x09F911029D74E35BD84156C5635688C0 % 2**32)
      init_rng, train_rng = jax.random.split(rng)
      params = net.init(init_rng, X_train[0], is_training=True)
      if exists("CGNN.params"):
          with open("CGNN.params", "rb") as f:
              params = pickle.load(f)
          print("Loaded model from CGNN.params")

      
      # Create the optimizer
      # Learning rate schedule: linear ramp-up and then constant
      num_epochs = 1000
      num_batches = len(X_train) // hp["batch_size"]
      ramp_up_epochs = 5  # Number of epochs to linearly ramp up the learning rate
      total_ramp_up_steps = ramp_up_epochs * num_batches
      lr_schedule = optax.linear_schedule(init_value=1e-5, 
                                          end_value =1e-3, 
                                          transition_steps=total_ramp_up_steps)

      # Optimizer
      optimizer = optax.noisy_sgd(learning_rate=lr_schedule)
      opt_state = optimizer.init(params)

      compute_loss_fn = jax.jit(functools.partial(loss_fn, net,label_types,label_nums))
      compute_accuracy_fn = jax.jit(functools.partial(accuracy_fn, net,label_types,label_nums))



      try:
          training_acc = ExponentialDecayWeighting(0.99)
          for epoch in range(num_epochs):
              for i in range(num_batches):
                  batch_rng = jax.random.fold_in(train_rng, i)
                  batch_start, batch_end = i * hp["batch_size"], (i + 1) * hp["batch_size"]
                  X_batch = X_train[batch_start:batch_end]
                  y_batch = y_train[batch_start:batch_end]

                  (loss, accs), grad = jax.value_and_grad(compute_loss_fn, has_aux=True)(params, rng, pad_graph_to_nearest_power_of_two(jraph.batch(X_batch)), y_batch)
                  updates, opt_state = optimizer.update(grad, opt_state)
                  params = optax.apply_updates(params, updates)
                  training_acc.add_accuracy(accs)
                  print("training accuracy: ", training_acc.get_weighted_average())
              

              # Save the training and validation loss - just validation, not implemented yet
      except KeyboardInterrupt:
          with open("CGNN.params", "wb") as f:
              pickle.dump(params, f)
          print("Keyboard interrupt, saving model")
      
      with open("CGNN.params", "wb") as f:
          pickle.dump(params, f)
      print("Done training")
