import sys, os, shutil
from tqdm import tqdm

# if not __name__ == '__main__':
#   os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

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



from includes import *
load_submodules() #this is a function in includes.py that loads all the submodules.


# hyperparameters
hp = {
    "layer_size" : None, # number of perceptrons in a layer. If None provided, automatically determined by the number of inputs
    "batch_size": 1000,
    "dropout_rate": 0.1,
    "num_proc": 12,#This is the number of processes used in preprocessing
    "atom_bin_sizes": (120, 80, 60, 40, 40, 40)
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


def processID(valid_material_id):
    print(valid_material_id)

    poscar = preprocessPoscar(valid_material_id)
    
    # Collect material inputs
    material_input_global =  [poscar_global.info(poscar) for poscar_global in poscar_globals]
    material_input_global += [global_input.info(valid_material_id) for global_input in global_inputs]
    material_input_atomic = [sum(items, []) for items in zip(*[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics])]


    # Collect material outputs
    material_output = []
    for func in range(len(global_outputs)):
        material_output.append(global_outputs[func].info(valid_material_id))

    atom_names = atom_names_in_poscar(poscar)


    atom_counts = Counter(atom_names)
    sorted_atoms = sorted(atom_counts.items(), key=lambda x: (-x[1], x[0]))
    ordering = {atom: j for j, (atom, _) in enumerate(sorted_atoms)}

    # Create a mapping between each sublist in `material_input_atomic` and its corresponding atom
    atom_pa_pairs = list(zip(atom_names, material_input_atomic))

    # Sort the sublists based on the atom frequencies stored in the `ordering` dictionary
    atom_pa_pairs_sorted = sorted(atom_pa_pairs, key=lambda x: ordering[x[0]])

    # Grab one set of atom properties of each atom type
    unique_atoms = {}
    for atom_string, atom_list in atom_pa_pairs_sorted:
        if atom_string not in unique_atoms:
            unique_atoms[atom_string] = atom_list
    unique_atom_tuples_sorted_by_count = [(atom_string, unique_atoms[atom_string], atom_counts[atom_string]) for atom_string in unique_atoms]

    # pad the sorted atom list to match the number of atom bins
    if len(unique_atom_tuples_sorted_by_count) > len(hp["atom_bin_sizes"]):
        return ("skipped", None, None, None)
    while len(unique_atom_tuples_sorted_by_count) < len(hp["atom_bin_sizes"]):
        unique_atom_tuples_sorted_by_count.append(("none", [0]*len(unique_atom_tuples_sorted_by_count[0][1]), 0))
    
    # create bins
    processed_atoms = []
    for i in range(len(unique_atom_tuples_sorted_by_count)):
        element = unique_atom_tuples_sorted_by_count[i]
        if element[2] > hp["atom_bin_sizes"][i]:
            return ("skipped", None, None, None)
        processed_atoms.append(
            [
                element[1],
                [1]*element[2] + [0]*(hp["atom_bin_sizes"][i]-element[2])
            ]
        )
    
    return (None, flatten(material_input_global),flatten(processed_atoms),flatten(material_output))


output_dim = 0#initialize before calling the network

if __name__ == '__main__':
    if not exists("NNN.pickle"):
        print("Preprocessing")

        #the atomic embeddings are assumed to be both input and available for all
        #poscar files. The global embeddings are assumed to not exist for all elements.
        #The poscar_globals is just the cell size. The others should be obvious.

        # Get the set of all materials with a POSCAR file available
        set_of_poscars = getSetOfPoscars()
        print("There are", len(set_of_poscars), "poscars total.")

        # Get the set of all materials with all selected properties (e.g. space group, band gap)
        common_valid_material_ids = set_of_poscars
        for component in global_inputs + global_outputs:
            common_valid_material_ids = set.intersection(common_valid_material_ids, component.valid_ids())

        print("There are", len(common_valid_material_ids), "materials with all inputs and outputs.")
        
        print("Loading data...")

        db_material_input_global = []
        db_material_input_atomic = []
        db_material_output = []
        skipped_atom_count = 0

        def accumulate_data(args):
            global db_material_input_global
            global db_material_input_atomic
            global db_material_output
            global skipped_atom_count
            
            if args[0] == "skipped":
                skipped_atom_count += 1
            else:
                db_material_input_global.append(args[1])
                db_material_input_atomic.append(args[2])
                db_material_output.append(args[3])


        go_types = [i.classifier() for i in global_outputs]

        # Grab the size of the first valid element in each global output
        go_nums = [len(i.info(list(i.valid_ids())[0])) for i in global_outputs] 

        pool = Pool(processes=hp["num_proc"])
        #Was going to batch the graphs, but padding them is incredibly memory intensive

        for index in common_valid_material_ids:
            pool.apply_async( processID, args=(index,) , callback=accumulate_data )

        pool.close()
        pool.join()

        print(skipped_atom_count, "poscars were skipped due to binning issues.")

        with open("NNN.pickle", "wb") as f:
            pickle.dump({
                "data": np.concatenate((np.array(db_material_input_atomic), np.array(db_material_input_global)), axis=1),
                "labels": np.array(db_material_output),
                "labels_types": go_types,
                "labels_nums": go_nums
            },f)


    with open("NNN.pickle", "rb") as f:
        print("Training")
        data = pickle.load(f)
        inputs = data["data"]
        labels = data["labels"]
        label_types = data["labels_types"]
        label_nums = data["labels_nums"]

        output_dim = sum(label_nums)

        print("loading ", len(inputs), "data points")


        # Split the data into training and validation sets
        X_train, y_train, X_val, y_val = partition_dataset(0.1, inputs, labels)

        # Automatically adjust the size of the NN if requested by the user (by setting "layer_size" to None)
        if not hp["layer_size"]:
            hp["layer_size"] = len(X_train[0])

        # Now, attempt to load the model NNN.params if it exists, otherwise init with haiku
        # Initialize the network
        net = hk.transform(net_fn)
        rng = jax.random.PRNGKey(0x4d696361684d756e64790a % 2**32)
        init_rng, train_rng = jax.random.split(rng)
        params = net.init(init_rng, jnp.array([X_train[0]]), is_training=True)
        if exists("NNN.params"):
            with open("NNN.params", "rb") as f:
                params = pickle.load(f)
            print("Loaded model from NNN.params")

        
        # Create the optimizer
        # Learning rate schedule: linear ramp-up and then constant
        num_epochs = 5000
        num_batches = X_train.shape[0] // hp["batch_size"]
        ramp_up_epochs = 100  # Number of epochs to linearly ramp up the learning rate
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
            for epoch in tqdm(range(num_epochs)):
                for valid_material_id in range(num_batches):
                    batch_rng = jax.random.fold_in(train_rng, valid_material_id)
                    batch_start, batch_end = valid_material_id * hp["batch_size"], (valid_material_id + 1) * hp["batch_size"]
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
