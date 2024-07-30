from includes import *
import os
import shutil

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
load_submodules()


# hyperparameters
hp = {
    "dropoutRate": 0.1,
    "max_atoms": 40,
    "batch_size": 512,
    "num_proc": 12,  # this is the number of processes used in preprocessing
}


def mab(x, y):
    head_count = 4
    key_count = 64
    query = jax.nn.leaky_relu(hk.linear(key_count * head_count)(x))
    key = jax.nn.leaky_relu(hk.linear(key_count * head_count)(y))
    value = jax.nn.leaky_relu(hk.linear(key_count * head_count)(y))
    attention_output = hk.multi_head_attention(
        num_heads=head_count, key_size=key_count, w_init_scale=1.0
    )(query, key, value)
    h = hk.layer_norm(axis=-1, create_scale=True, create_offset=True)(
        query + attention_output
    )
    ret = hk.layer_norm(axis=-1, create_scale=True, create_offset=True)(
        h + hk.nets.mlp([256, 128, 128, h.shape[-1]], activation=jax.nn.leaky_relu)(h)
    )
    return ret


def sab(x):
    return mab(x, x)


def isab(x, param_name, num_inducing_points=16):
    inducing_points = hk.get_parameter(
        param_name, shape=(num_inducing_points, 256), init=jnp.zeros
    )
    h = mab(inducing_points, x)
    return mab(x, h)


def encoder(x):
    x = sab(x)
    return x


def decoder(z):
    z = hk.sequential([hk.linear(256), jax.nn.leaky_relu])(
        z
    )  # check rep dim - prob not needed
    num_seeds = 12  # apply sab after if you ever need more vectors out the end
    seed_vectors = hk.get_parameter(
        "seed_vec", shape=[num_seeds, z.shape[-1]], init=hk.initializers.random_normal()
    )
    z = mab(seed_vectors, z)
    z = sab(z)
    z = hk.sequential(
        [
            hk.linear(256),
            jax.nn.leaky_relu,
            hk.linear(128),
            jax.nn.leaky_relu,
            hk.linear(64),
            jax.nn.leaky_relu,
            hk.linear(output_dim),
        ]
    )(z)
    return z[0]  # implicit global var to avoid passing static_args


def set_transformer_single(x):
    # encoder
    z = encoder(x)

    # decoder
    output = decoder(z)
    return output


def set_transformer(batch, is_training=False, dropout_rate=0):
    # reshape input to (batch_size, n, d_x)
    batch_atoms = batch[0]
    batch_global = batch[1]

    # now, batch_atoms is of the form (batch.shape[0], 60, -1) i want to take
    # the batch_global of the form (batch.shape[0], -1), tile it out and
    # attach it to each array

    batch_g = jnp.tile(batch_global[:, none, :], (1, hp["max_atoms"], 1))
    reshaped_batch = jnp.concatenate((batch_atoms, batch_g), axis=2)

    # apply set_transformer_single to each example in the batch using jax.vmap
    output = jax.vmap(set_transformer_single)(reshaped_batch)

    return output


def net_fn(batch, is_training=False, dropout_rate=0):
    return set_transformer(batch)


def process_id(valid_material_id):
    poscar = preprocess_poscar(valid_material_id)

    material_input_global = [
        poscar_global.info(poscar) for poscar_global in poscar_globals
    ]
    material_input_global += [
        global_input.info(valid_material_id) for global_input in global_inputs
    ]
    material_input_atomic = np.array(
        [
            sum(items, [])
            for items in zip(
                *[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics]
            )
        ]
    )

    # collect material outputs
    material_output = []
    for func in range(len(global_outputs)):
        material_output.append(global_outputs[func].info(valid_material_id))

    atom_names = atom_names_in_poscar(poscar)

    if len(atom_names) > hp["max_atoms"]:
        return ("Large", none, none, none)

    material_input_atomic = np.concatenate(
        (
            material_input_atomic,
            np.zeros(
                (hp["max_atoms"] - len(atom_names), material_input_atomic.shape[1])
            ),
        )
    )

    return (
        none,
        np.array(flatten(material_input_global)),
        material_input_atomic,
        np.array(flatten(material_output)),
    )


output_dim = 0  # initialize before calling the network

if __name__ == "__main__":
    if not exists("CANN.pickle"):
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

        db_material_input_global = []
        db_material_input_atomic = []
        db_material_output = []
        skipped_atom_count = 0

        def accumulate_data(args):
            global db_material_input_global
            global db_material_input_atomic
            global db_material_output
            global skipped_atom_count

            if args[0] == "Large":
                skipped_atom_count += 1
            else:
                db_material_input_global.append(args[1])
                db_material_input_atomic.append(args[2])
                db_material_output.append(args[3])

        go_types = [i.classifier() for i in global_outputs]

        # grab the size of the first valid element in each global output
        go_nums = [len(i.info(list(i.valid_ids())[0])) for i in global_outputs]

        Pool = Pool(processes=hp["num_proc"])
        # was going to batch the graphs, but padding them is incredibly memory
        # intensive

        for index in common_valid_material_ids:
            Pool.apply_async(process_id, args=(index,), callback=accumulate_data)

        Pool.close()
        Pool.join()

        print(skipped_atom_count, "poscars were skipped due to being too large.")

        with open("CANN.pickle", "wb") as f:
            pickle.dump(
                {
                    # do local then global data, so i can reconfigure cann without
                    # having to preprocess again
                    "data": (
                        np.array(db_material_input_atomic),
                        np.array(db_material_input_global),
                    ),
                    "labels": np.array(db_material_output),
                    "labels_types": go_types,
                    "labels_nums": go_nums,
                },
                f,
            )

    with open("CANN.pickle", "rb") as f:
        data = pickle.load(f)
        inputs = data["data"][0]
        inputs_global = data["data"][1]
        labels = data["labels"]
        label_types = data["labels_types"]
        label_nums = data["labels_nums"]

        output_dim = sum(label_nums)

        # split the data into training and validation sets
        # x_train, y_train, add_train, x_val, y_val, add_val = partition_dataset(0.4, data, labels, additional_data)
        x_train, globals_train, y_train, x_val, globals_val, y_val = partition_dataset(
            0.1, inputs, inputs_global, labels
        )

        # now, attempt to load the model cann.params if it exists, otherwise init with haiku
        # initialize the network
        net = hk.transform(net_fn)
        rng = jax.random.prng_key(0x09_f911029_d74_e35_bd84156_c5635688_c0 % 2**32)
        init_rng, train_rng = jax.random.split(rng)
        params = net.init(
            init_rng,
            (jnp.array([x_train[0]]), jnp.array([globals_train[0]])),
            is_training=True,
        )
        if exists("CANN.params"):
            with open("CANN.params", "rb") as f:
                params = pickle.load(f)
            print("Loaded model from CANN.params")

        # create the optimizer
        # learning rate schedule: linear ramp-up and then constant
        num_epochs = 1000
        num_batches = x_train.shape[0] // hp["batch_size"]
        ramp_up_epochs = 50  # number of epochs to linearly ramp up the learning rate
        total_ramp_up_steps = ramp_up_epochs * num_batches
        lr_schedule = optax.linear_schedule(
            init_value=1e-3, end_value=1e-3, transition_steps=total_ramp_up_steps
        )

        # optimizer
        optimizer = optax.noisy_sgd(learning_rate=lr_schedule)
        opt_state = optimizer.init(params)

        compute_loss_fn = jax.jit(
            functools.partial(loss_fn, net, label_types, label_nums)
        )
        compute_accuracy_fn = jax.jit(
            functools.partial(accuracy_fn, net, label_types, label_nums)
        )

        try:
            training_acc = ExponentialDecayWeighting(0.9)
            for epoch in range(num_epochs):
                for i in range(num_batches):
                    batch_rng = jax.random.fold_in(train_rng, i)
                    batch_start, batch_end = (
                        i * hp["batch_size"],
                        (i + 1) * hp["batch_size"],
                    )
                    x_batch = x_train[batch_start:batch_end]
                    x_batch_global = globals_train[batch_start:batch_end]
                    y_batch = y_train[batch_start:batch_end]
                    # print all the types for debug purposes:

                    (loss, accs), grad = jax.value_and_grad(
                        compute_loss_fn, has_aux=True
                    )(params, rng, (x_batch, x_batch_global), y_batch)
                    updates, opt_state = optimizer.update(grad, opt_state)
                    params = optax.apply_updates(params, updates)

                    training_acc.add_accuracy(accs)
                    if i % 10 == 0:
                        print(training_acc.get_weighted_average())

                # targets always has a shape - no funky stuff
                num_samples = jnp.array(y_val).shape[0]
                # ceiling division
                val_num_batches = -(-num_samples // hp["batch_size"])

                batch_accuracies = []

                for batch_idx in range(val_num_batches):
                    start_idx = batch_idx * hp["batch_size"]
                    end_idx = min((batch_idx + 1) * hp["batch_size"], num_samples)

                    batch_accuracy = compute_accuracy_fn(
                        params,
                        rng,
                        (x_val[start_idx:end_idx], globals_val[start_idx:end_idx]),
                        y_val[start_idx:end_idx],
                    )
                    batch_accuracies.append(batch_accuracy)

                # stack the batch accuracies and take the mean
                batch_accuracies = jnp.stack(batch_accuracies)
                mean_accuracy = jnp.mean(batch_accuracies, axis=0)

                print(f"Epoch {epoch}, Validation accuracy: {mean_accuracy}")
        except keyboard_interrupt:
            with open("CANN.params", "wb") as f:
                pickle.dump(params, f)
            print("Keyboard interrupt, saving model")

        with open("CANN.params", "wb") as f:
            pickle.dump(params, f)
        print("Done training")
