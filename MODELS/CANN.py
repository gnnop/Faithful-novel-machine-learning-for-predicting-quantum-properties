import sys, os, shutil

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
copy_file_to_directories('../includes.py', target_directories)


#Now the rest of the file.

from includes import *
load_submodules() #this is a function in includes.py that loads all the submodules.


# hyperparameters
hp = {
    "dropoutRate": 0.1,
    "max_atoms": 40,
    "batch_size": 64,
}


def MAB(X, Y, is_self_attention=False):
  if is_self_attention:
    Y = X

  attention_output = hk.MultiHeadAttention(num_heads=8, key_size=128, w_init_scale=1.0, model_size=X.shape[-1])(X, Y, Y)
  H = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(X + hk.Linear(X.shape[-1])(attention_output))
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(H + hk.nets.MLP([256, 128, 128, H.shape[-1]], activation=jax.nn.leaky_relu)(H))


def SAB(X):
  return MAB(X, X, is_self_attention=True)

def ISAB(X,param_name, num_inducing_points=16):
  inducing_points = hk.get_parameter(param_name, shape=(num_inducing_points, X.shape[-1]), init=jnp.zeros)
  H = MAB(inducing_points, X)
  return MAB(X, H, is_self_attention=False)

def PMA(Z, param_name):
  num_seeds=1 #apply SAB after if you ever need more vectors out the end
  seed_vectors = hk.get_parameter(param_name, shape=[num_seeds, Z.shape[-1]], init=hk.initializers.RandomNormal())
  return MAB(seed_vectors, hk.Linear(Z.shape[-1])(Z), is_self_attention=False)

def encoder(X):
  X = ISAB(ISAB(ISAB(X, "ind1"), "ind2"), "ind3")
  return X

def decoder(Z):
  #Z = hk.nets.MLP([512, 256, 64], activation=jax.nn.leaky_relu)(Z)#check rep dim - prob not needed
  Z = PMA(Z, "seed_vec")#SAB(PMA(Z))
  Z = hk.nets.MLP([512, 256, 64], activation=jax.nn.leaky_relu)(Z)
  return hk.Linear(output_dim)(Z)#implicit global var to avoid passing static_args

@jax.vmap
def set_transformer_single(X):
  # Encoder
  Z = encoder(X)

  # Decoder
  output = decoder(Z)

  return output

def set_transformer(batch, is_training=False, dropout_rate=0):
  # Reshape input to (batch_size, n, d_x)
  batch_atoms = batch[0]
  batch_global = batch[1]

  #Now, batch_atoms is of the form (batch.shape[0], 60, -1) I want to take the batch_global of the form (batch.shape[0], -1), tile it out and attach it to each array

  batch_g = jnp.tile(batch_global[:, None, :], (1, hp["max_atoms"], 1))
  reshaped_batch = jnp.concatenate((batch_atoms, batch_g), axis=2)

  # Apply set_transformer_single to each example in the batch using jax.vmap
  output = set_transformer_single(reshaped_batch)

  return output


def net_fn(batch, is_training=False, dropout_rate=0):
  return set_transformer(batch, output_dim)

output_dim = 0#initialize before calling the network

if not exists("CANN.pickle"):

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




    pi = []
    pa = []
    go = []
    go_types = [i.classifier() for i in global_outputs]
    go_nums = [len(i.info(str(1))) for i in global_outputs]

    ii = 0
    for i in ids:

        poscar = preprocessPoscar(i)

        #fixed dimensional
        pi_ = flatten([poscar_global.info(poscar) for poscar_global in poscar_globals] + [global_input.info(i) for global_input in global_inputs])

        #fixed dimensional
        go_ = []
        for func in range(len(global_outputs)):
            go_.append(global_outputs[func].info(i))

        atom_names = atom_names_in_poscar(poscar)

        if len(atom_names) > hp["max_atoms"]:
            ii += 1
            continue#store this
        
        #variable dimensional
        #Note that we interleave the atomics
        pa_ = np.array([sum(items, []) for items in zip(*[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics])])
        pa_ = np.concatenate((pa_, np.zeros((hp["max_atoms"] - len(atom_names), pa_.shape[1]))))
        

        pi.append(pi_)
        pa.append(pa_)
        go.append(flatten(go_))

    print(ii, "poscars were skipped due to having more than", hp["max_atoms"], "atoms.")

    with open("CANN.pickle", "wb") as f:
        pickle.dump({
            "data": (np.array(pa), np.array(pi)),#do local then global data, so I can reconfigure CANN without having to preprocess again
            "labels": np.array(go),
            "labels_types": go_types,
            "labels_nums": go_nums
        },f)


with open("CANN.pickle", "rb") as f:
    data = pickle.load(f)
    inputs = data["data"][0]
    inputs_global = data["data"][1]
    labels = data["labels"]
    label_types = data["labels_types"]
    label_nums = data["labels_nums"]

    output_dim = sum(label_nums)


    # Split the data into training and validation sets
    ##X_train, y_train, add_train, X_val, y_val, add_val = partition_dataset(0.4, data, labels, additional_data)
    X_train, globals_train, y_train, X_val, globals_val, y_val = partition_dataset(0.1, inputs, inputs_global, labels)

    #Now, attempt to load the model NNN.params if it exists, otherwise init with haiku
    # Initialize the network
    net = hk.transform(net_fn)
    rng = jax.random.PRNGKey(0x09F911029D74E35BD84156C5635688C0 % 2**32)
    init_rng, train_rng = jax.random.split(rng)
    params = net.init(init_rng, (jnp.array([X_train[0]]),jnp.array([globals_train[0]])), is_training=True)
    if exists("CANN.params"):
        with open("CANN.params", "rb") as f:
            params = pickle.load(f)
        print("Loaded model from NNN.params")

    
    # Create the optimizer
    # Learning rate schedule: linear ramp-up and then constant
    num_epochs = 1000
    num_batches = X_train.shape[0] // hp["batch_size"]
    ramp_up_epochs = 50  # Number of epochs to linearly ramp up the learning rate
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
        training_acc = ExponentialDecayWeighting(0.9)
        for epoch in range(num_epochs):
            for i in range(num_batches):
                batch_rng = jax.random.fold_in(train_rng, i)
                batch_start, batch_end = i * hp["batch_size"], (i + 1) * hp["batch_size"]
                X_batch = X_train[batch_start:batch_end]
                X_batch_global = globals_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                #print all the types for debug purposes:

                (loss, accs), grad = jax.value_and_grad(compute_loss_fn, has_aux=True)(params, rng, (X_batch,X_batch_global), y_batch)
                updates, opt_state = optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, updates)

                training_acc.add_accuracy(accs)
                if i % 10 == 0:
                  print(training_acc.get_weighted_average())

                
            

            # Save the training and validation loss - for uniformity, we need to make sure this is always the same. Also, for some networks, the following needs to be shrunk
            val_accuracy = compute_accuracy_fn(params, batch_rng, (X_val, globals_val), y_val)
            print(f"Epoch {epoch}, Validation accuracy: {val_accuracy}")
    except KeyboardInterrupt:
        with open("NNN.params", "wb") as f:
            pickle.dump(params, f)
        print("Keyboard interrupt, saving model")
    
    with open("NNN.params", "wb") as f:
        pickle.dump(params, f)
    print("Done training")
