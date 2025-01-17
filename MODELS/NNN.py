import sys, os, shutil

if not __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "False"

# standard thing that makes sure all dependencies resolve.


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


# define the relative paths of the target directories
target_directories = [".", "input", "output"]
if __name__ == "__main__":
    copy_file_to_directories("../includes.py", target_directories)


# now the rest of the file.

# this is a function in includes.py that loads all the submodules.
from includes import *
load_submodules()


# hyperparameters
hp = {
    "layer_size": 1000,  # number of perceptrons in a layer. if None provided, automatically determined by the number of inputs
    "batch_size": 1000,  # if None, the size of the dataset
    "dropout_rate": 0.1,
    "max_epochs": 10000,
    "num_proc": 12,  # this is the number of processes used in preprocessing
    "atom_bin_sizes": (40, 24, 16, 12, 8, 6), #These are the bins for each atomic type
}


# define the neural network
def net_fn(batch, is_training=False):
    mlp = hk.Sequential(
        [
            # fully connected layer with dropout
            hk.Linear(hp["layer_size"]),
            jax.nn.relu,
            # lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropout_rate"]) if is_training else x,
            # fully connected layer with dropout
             hk.Linear(hp["layer_size"]), jax.nn.relu,
            # lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropout_rate"]) if is_training else x,
            # fully connected layer with dropout
             hk.Linear(hp["layer_size"]), jax.nn.relu,
            # lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropout_rate"]) if is_training else x,
            # # fully connected layer with dropout
            # hk.Linear(output_dim), jax.nn.relu,
            # # lambda x: hk.dropout(rng=hk.next_rng_key(), x=x, rate=hp["dropout_rate"]) if is_training else x,
            hk.Linear(output_dim),  # assuming full set of categories
        ]
    )
    return mlp(batch)


def process_id(valid_material_id):
    poscar = preprocess_poscar(valid_material_id)

    # collect material inputs
    material_input_global = [
        poscar_global.info(poscar) for poscar_global in poscar_globals
    ] + [1]
    material_input_global += [
        global_input.info(valid_material_id) for global_input in global_inputs
    ]
    material_input_atomic = [
        sum(items, []) + [1]
        for items in zip(
            *[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics]
        )
    ]

    # collect material outputs
    material_output = []
    for func in range(len(global_outputs)):
        material_output.append(global_outputs[func].info(valid_material_id))

    # assign all data specific to individual atoms into bins.
    # a list of strings: the chemical symbol of each atom in the poscar file
    # in order of appearance
    atom_names = atom_names_in_poscar(poscar)
    atom_counts = Counter(atom_names)
    # a list of 2-tuples associating elements with their counts in the poscar
    # file: (chemical symbol, frequency in poscar). sorted by frequency in
    # descending order.
    sorted_atoms = sorted(atom_counts.items(), key=lambda x: (-x[1], x[0]))
    # a dictionary that helps decide which atoms belong into which bins. the
    # key is a chemical symbol, and the value is the index of a bin. to assign
    # an atom to a bin, we look up its chemical symbol in this dictionary. the
    # index of the bin is the value returned by the dictionary.
    ordering = {atom: j for j, (atom, _) in enumerate(sorted_atoms)}
    # a list of 2-tuples associating atomic data with its chemical symbol:
    # (chemical symbol of an atom in poscar, preprocessed data generated by
    # the atomic inputs). one entry exists for each atom and is in the order
    # of the poscar file.
    poscar_atomic_pairs = list(zip(atom_names, material_input_atomic))
    # sorts items (atoms) of poscar_atomic_pairs by the frequency of its
    # chemical symbol in the poscar file. that way we construct the bins in
    # order of size naturally by iterating through the elements in this list.
    poscar_atomic_pairs = sorted(poscar_atomic_pairs, key=lambda x: ordering[x[0]])
    # grab one set of atom properties of each atom type
    unique_atoms = {}
    for atom_string, atom_list in poscar_atomic_pairs:
        if atom_string not in unique_atoms:
            unique_atoms[atom_string] = atom_list
    unique_atom_tuples_sorted_by_count = [
        (atom_string, unique_atoms[atom_string], atom_counts[atom_string])
        for atom_string in unique_atoms
    ]
    # now unique_atom_tuples_sorted_by_count is a list of 3-tuples of unique elements whose structure is as follows:
    # (chemical symbol, one example atom of that type, number of atoms of that type in the poscar file)
    # the items in this list are sorted by the frequency (3rd entry of each tuple) in decending value
    # pad the sorted atom list to match the number of atom bins
    if len(unique_atom_tuples_sorted_by_count) > len(hp["atom_bin_sizes"]):
        return ("skipped", None, None, None)
    while len(unique_atom_tuples_sorted_by_count) < len(hp["atom_bin_sizes"]):
        unique_atom_tuples_sorted_by_count.append(
            ("None", [0] * len(unique_atom_tuples_sorted_by_count[0][1]), 0)
        )
    # create bins
    processed_atoms = []
    for i in range(len(unique_atom_tuples_sorted_by_count)):
        element = unique_atom_tuples_sorted_by_count[i]
        if element[2] > hp["atom_bin_sizes"][i]:
            return ("skipped", None, None, None)
        processed_atoms.append(
            [
                element[1],
                [1] * element[2] + [0] * (hp["atom_bin_sizes"][i] - element[2]),
            ]
        )
    # now processed_atoms is a list of lists of lists.
    """
    Bin 0: element 0
        2-item list
            Item 1: an example atom, preprocessed by all poscar_atomic_* scripts in the input/ folder
            Item 2: 1 if the atom exists, 0 if it does not. Length = hp["atom_bin_sizes"][0]
    Bin 1: element 1
        2-item list
            Item 1: an example atom, preprocessed by all poscar_atomic_* scripts in the input/ folder
            Item 2: 1 if the atom exists, 0 if it does not. Length = hp["atom_bin_sizes"][1]
    ...
    Bin 5: element 5
        2-item list
            Item 1: an example atom, preprocessed by all poscar_atomic_* scripts in the input/ folder
            Item 2: 1 if the atom exists, 0 if it does not. Length = hp["atom_bin_sizes"][5]

    """
    # this concludes atom preprocessing.
    # material_input_global and material_output need no further preprocessing
    # and are returned as is.
    return (
        None,
        flatten(material_input_global),
        flatten(processed_atoms),
        flatten(material_output),
    )


output_dim = 0  # initialize before calling the network

if __name__ == "__main__":
    if not exists("NNN.pickle"):
        print("Preprocessing")

        print(
            f"""
There are {len(global_inputs)}  global inputs,
          {len(global_outputs)} global outputs,
          {len(poscar_globals)} POSCAR global inputs, and
          {len(poscar_atomics)} POSCAR atomic inputs
        """
        )

        # the atomic embeddings are assumed to be both input and available for all
        # poscar files. the global embeddings are assumed to not exist for all elements.
        # the poscar_globals is just the cell size. the others should be
        # obvious.

        # get the set of all materials with a poscar file available
        set_of_poscars = get_set_of_poscars()
        print("There are", len(set_of_poscars), "poscars total.")

        # get the set of all materials with all selected properties (e.g. space
        # group, band gap)
        common_valid_material_ids = set_of_poscars
        for component in global_inputs + global_outputs:
            common_valid_material_ids = set.intersection(
                common_valid_material_ids, component.valid_ids()
            )

        print(
            "There are",
            len(common_valid_material_ids),
            "materials with all inputs and outputs.",
        )

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

        # grab the size of the first valid element in each global output
        go_nums = [len(i.info(list(i.valid_ids())[0])) for i in global_outputs]

        pool = Pool(processes=hp["num_proc"])
        # was going to batch the graphs, but padding them is incredibly memory
        # intensive

        for index in common_valid_material_ids:
            pool.apply_async(process_id, args=(index,), callback=accumulate_data)

        pool.close()
        pool.join()

        print(skipped_atom_count, "poscars were skipped due to binning issues.")

        d = {
            "data": np.concatenate(
                (
                    np.array(db_material_input_atomic),
                    np.array(db_material_input_global),
                ),
                axis=1,
            ),
            "labels": np.array(db_material_output),
            "labels_types": go_types,
            "labels_nums": go_nums,
        }

        with open("NNN.pickle", "wb") as f:
            pickle.dump(d, f)

    with open("NNN.pickle", "rb") as f:
        print("Training")
        data = pickle.load(f)
        inputs = data["data"]
        labels = data["labels"]
        label_types = data["labels_types"]
        label_nums = data["labels_nums"]

        output_dim = sum(label_nums)

        print("loaded ", len(inputs), "data points")
        # split the data into training and validation sets
        X_train, y_train, X_val, y_val = partition_dataset(0.1, inputs, labels)
        print(f"Number of elements in an input:  {len(X_train[0])}")
        print(f"Number of elements in an output: {len(y_train[0])}")

        # automatically adjust the size of the nn if requested by the user (by
        # setting "layer_size" to None)
        if not hp["layer_size"]:
            hp["layer_size"] = len(X_train[0])

        # now, attempt to load the model nnn.params if it exists, otherwise init with haiku
        # initialize the network
        net = hk.transform_with_state(net_fn)
        rng = jax.random.PRNGKey(0x4_d696361684_d756_e64790_a % 2**32)
        init_rng, train_rng = jax.random.split(rng)
        params, state = net.init(init_rng, jnp.array([X_train[0]]), is_training=True)
        if exists("NNN.params"):
            with open("NNN.params", "rb") as f:
                params = pickle.load(f)
            print("Loaded model from NNN.params")

        # create the optimizer
        # learning rate schedule: linear ramp-up and then constant
        if not hp["batch_size"]:
            hp["batch_size"] = len(X_train)

        # check memory usage
        batch_memory = hp["batch_size"] * len(X_train[0]) * 4
        print(f"Batch memory: {batch_memory/1e9:.01} GB")

        num_batches = X_train.shape[0] // hp["batch_size"]
        ramp_up_epochs = 500  # number of epochs to linearly ramp up the learning rate
        total_ramp_up_steps = ramp_up_epochs * num_batches
        opt_init, opt_update = optax.adam(3e-4)
        opt_state = opt_init(params)

        compute_loss_fn = jax.jit(
            functools.partial(loss_fn, net, label_types, label_nums)
        )
        compute_accuracy_fn = jax.jit(
            functools.partial(accuracy_fn, net, label_types, label_nums)
        )
        val_and_grad = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))

        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots(figsize=(10, 8))
        line, = ax.plot([1, 2], [1, 2])
        # setting x-axis label and y-axis label
        plt.xlabel("batches")
        plt.ylabel("accuracy")

        try:
            iii = 0
            training_acc = ExponentialDecayWeighting(0.99)
            training_res = []
            for epoch in range(hp["max_epochs"]):
                train_rng = jax.random.split(train_rng, num=1)[0]
                shuffled_indices = jax.random.permutation(train_rng, num_batches)
                for valid_material_id in range(num_batches):
                    batch_rng = jax.random.fold_in(train_rng, valid_material_id)
                    batch_start, batch_end = (
                        shuffled_indices[valid_material_id] * hp["batch_size"],
                        (shuffled_indices[valid_material_id] + 1) * hp["batch_size"],
                    )
                    x_batch = X_train[batch_start:batch_end]
                    y_batch = y_train[batch_start:batch_end]
                    # print all the types for debug purposes:

                    (loss, (accs, state)), grad = val_and_grad(params,state, rng, x_batch, y_batch)
                    updates, opt_state = opt_update(grad, opt_state)
                    params = optax.apply_updates(params, updates)
                    training_acc.add_accuracy(accs)
                    training_res.append(training_acc.get_weighted_average())
                    iii += 1

                    if iii % 10 == 0:#adjust per your feelings
                        line.set_xdata(np.arange(len(training_res)))
                        line.set_ydata(np.array(training_res))
                        ax.relim()
                        ax.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                # save the testing loss
                num_batches = X_val.shape[0] // hp["batch_size"]
                batch_accuracies = []
                for batch in range(num_batches):
                    batch_rng = jax.random.fold_in(train_rng, batch)
                    batch_start, batch_end = (
                        batch * hp["batch_size"],
                        (batch + 1) * hp["batch_size"],
                    )
                    x_batch = X_val[batch_start:batch_end]
                    y_batch = y_val[batch_start:batch_end]
                    # print all the types for debug purposes:
                    batch_accuracy = compute_accuracy_fn(params,state, batch_rng, X_val, y_val)
                    batch_accuracies.append(batch_accuracy)
                
                batch_accuracies = jnp.stack(batch_accuracies)
                mean_accuracy = jnp.mean(batch_accuracies)
                print(f"Epoch {epoch}, Validation accuracy: {mean_accuracy}")
        except KeyboardInterrupt:
            with open("NNN.params", "wb") as f:
                pickle.dump(params, f)
            print("Keyboard interrupt, saving model")

        with open("NNN.params", "wb") as f:
            pickle.dump(params, f)
        
        plt.ioff()  # Turn off interactive mode
        plt.savefig('training_loss_NNN.png')
        plt.close()
        print("Done training")
