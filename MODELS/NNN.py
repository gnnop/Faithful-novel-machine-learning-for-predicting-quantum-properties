import sys, os, shutil

os.chdir(os.path.dirname(os.path.realpath(__file__)))

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
target_directories = ['input', 'output']
copy_file_to_directories('includes.py', target_directories)


#Now the rest of the file.

from includes import *
load_submodules() #this is a function in includes.py that loads all the submodules.


# hyperparameters
hp = {
    "layer_size" : None, # number of perceptrons in a layer. If None provided, automatically determined by the number of inputs
    "max_atoms": 40,
    "batch_size": 1000,
    "dropout_rate": 0.1,
}



# Define the neural network
def net_fn(batch, is_training=False):
    mlp = hk.Sequential([
        # fully connected layer with dropout
        hk.Linear(hp["layer_size"]), jax.nn.relu,
        lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropout_rate"]) if is_training else x,  

        # fully connected layer with dropout
        hk.Linear(hp["layer_size"]), jax.nn.relu, 
        lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropout_rate"]) if is_training else x,  

        # fully connected layer with dropout
        hk.Linear(hp["layer_size"]), jax.nn.relu,
        lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropout_rate"]) if is_training else x,  
        
        hk.Linear(output_dim)  # Assuming full set of categories
    ])
    return mlp(batch)




output_dim = 0#initialize before calling the network

if not exists("NNN.pickle"):

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

    # Grab the size of the first valid element in each global output
    go_nums = [len(i.info(list(i.valid_ids())[0])) for i in global_outputs] 

    ii = 0
    for i in ids:

        poscar = preprocessPoscar(i)

        #fixed dimensional
        pi_ = [poscar_global.info(poscar) for poscar_global in poscar_globals] + [global_input.info(i) for global_input in global_inputs]

        #variable dimensional
        #Note that we interleave the atomics
        pa_ = [sum(items, []) for items in zip(*[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics])]


        #fixed dimensional
        go_ = []
        for func in range(len(global_outputs)):
            go_.append(global_outputs[func].info(i))

        atom_names = atom_names_in_poscar(poscar)

        if len(atom_names) > hp["max_atoms"]:
            ii += 1
            continue#store this


        atom_counts = Counter(atom_names)
        sorted_atoms = sorted(atom_counts.items(), key=lambda x: (-x[1], x[0]))
        ordering = {atom: j for j, (atom, _) in enumerate(sorted_atoms)}

        # Create a mapping between each sublist in `pa_` and its corresponding atom
        atom_pa_pairs = list(zip(atom_names, pa_))

        # Sort the sublists based on the atom frequencies stored in the `ordering` dictionary
        atom_pa_pairs_sorted = sorted(atom_pa_pairs, key=lambda x: ordering[x[0]])

        # Extract the sorted pa_
        pa_sorted = [pair[1] for pair in atom_pa_pairs_sorted]

        # Now, buff `pa_sorted` out with sublists of the same length as all the other sublists in it until its length max_atoms
        pa_sorted = pa_sorted + [[0. for _ in range(len(pa_sorted[0]))] for _ in range(hp["max_atoms"] - len(pa_sorted))]


        pi.append(flatten(pi_))
        pa.append(flatten(pa_sorted))
        go.append(flatten(go_))

    print(ii, "poscars were skipped due to having more than", hp["max_atoms"], "atoms.")

    with open("NNN.pickle", "wb") as f:
        pickle.dump({
            "data": np.concatenate((np.array(pa), np.array(pi)), axis=1),
            "labels": np.array(go),
            "labels_types": go_types,
            "labels_nums": go_nums
        },f)


with open("NNN.pickle", "rb") as f:
    data = pickle.load(f)
    inputs = data["data"]
    labels = data["labels"]
    label_types = data["labels_types"]
    label_nums = data["labels_nums"]

    output_dim = sum(label_nums)


    # Split the data into training and validation sets
    X_train, y_train, X_val, y_val = partition_dataset(0.1, inputs, labels)

    # Automatically adjust the size of the NN if requested by the user (by setting "layer_size" to None)
    if not hp["layer_size"]:
        hp["layer_size"] = len(X_train[0])

    print(len(X_train))
    print(len(X_train[0]))
    print(len(y_train))
    print(len(y_train[0]))
    print(X_train[0])
    print(y_train[0])

    # Now, attempt to load the model NNN.params if it exists, otherwise init with haiku
    # Initialize the network
    net = hk.transform(net_fn)
    rng = jax.random.PRNGKey(0x4d696361684d756e64790a % 2**32)
    init_rng, train_rng = jax.random.split(rng)
    params = net.init(init_rng, X_train[0], is_training=True)
    if exists("NNN.params"):
        with open("NNN.params", "rb") as f:
            params = pickle.load(f)
        print("Loaded model from NNN.params")

    
    # Create the optimizer
    # Learning rate schedule: linear ramp-up and then constant
    num_epochs = 5000
    num_batches = X_train.shape[0] // hp["batch_size"]
    ramp_up_epochs = 500  # Number of epochs to linearly ramp up the learning rate
    total_ramp_up_steps = ramp_up_epochs * num_batches
    lr_schedule = optax.linear_schedule(init_value=1e-6, 
                                        end_value =1e-5, 
                                        transition_steps=total_ramp_up_steps)

    # Optimizer
    optimizer = optax.noisy_sgd(learning_rate=lr_schedule)
    opt_state = optimizer.init(params)

    compute_loss_fn = jax.jit(functools.partial(loss_fn, net,label_types,label_nums))
    compute_accuracy_fn = jax.jit(functools.partial(accuracy_fn, net,label_types,label_nums))



    try:
        for epoch in range(num_epochs):
            for i in range(num_batches):
                batch_rng = jax.random.fold_in(train_rng, i)
                batch_start, batch_end = i * hp["batch_size"], (i + 1) * hp["batch_size"]
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]
                #print all the types for debug purposes:

                (loss, accs), grad = jax.value_and_grad(compute_loss_fn, has_aux=True)(params, rng, X_batch, y_batch)
                updates, opt_state = optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, updates)
            

            # Save the training and validation loss
            train_accuracy = compute_accuracy_fn(params, batch_rng, X_train, y_train)
            val_accuracy = compute_accuracy_fn(params, batch_rng, X_val, y_val)
            print(f"Epoch {epoch}, Training accuracy: {train_accuracy}, Validation accuracy: {val_accuracy}")
    except KeyboardInterrupt:
        with open("NNN.params", "wb") as f:
            pickle.dump(params, f)
        print("Keyboard interrupt, saving model")
    
    with open("NNN.params", "wb") as f:
        pickle.dump(params, f)
    print("Done training")
