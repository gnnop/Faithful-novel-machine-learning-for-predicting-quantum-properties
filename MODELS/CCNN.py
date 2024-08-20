import os
import shutil

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
    "dropoutRate": 0.1,
    "max_atoms": 40,
    "batch_size": 32,
    # the input vector is a 3d lattice of size max_dims x max_dims x max_dims of
    # atoms, represented as vectors of size max_rep.
    "maxDims": 48,
    # it is assumed that all atoms have the same size. this is the relative
    # radius of each atom in lattice units.
    "conversionFactor": 1.45,
    "num_proc": 8,
}

hp["centre"] = np.array([hp["maxDims"] / 2, hp["maxDims"] / 2, hp["maxDims"] / 2])
# the last entry is dynamically created before network operation
hp["dims"] = ([hp["maxDims"], hp["maxDims"], hp["maxDims"]], 1)


# using the includes file loss functions, this is necessary for it.
def net_fn(batch_input, is_training=True, dropout_rate=0.1):
    batch_conv, batch_global = batch_input
    
    def conv_block(x, channels, kernel_size=3, stride=1, use_bn=True):
        x = hk.Conv3D(output_channels=channels, kernel_shape=kernel_size, stride=stride)(x)
        if use_bn:
            x = hk.BatchNorm(True, True, 0.99)(x, is_training=is_training)
        return jax.nn.leaky_relu(x)

    # Process convolutional data
    batch_conv = conv_block(batch_conv, kernel_size=20, channels=6, stride=3, use_bn=False)
    batch_conv = conv_block(batch_conv, kernel_size=20, channels=6, stride=3, use_bn=False)
    batch_conv = conv_block(batch_conv, channels=40, stride=3, use_bn=False)
    batch_conv = conv_block(batch_conv, channels=102, stride=3, use_bn=False)
    batch_conv = conv_block(batch_conv, channels=306, stride=3, use_bn=False)
    batch_conv = conv_block(batch_conv, channels=1024, stride=3, use_bn=False)
    batch_conv = jnp.squeeze(batch_conv, axis=(1, 2, 3))

    # Global average pooling
    #x = jnp.mean(x, axis=(1, 2, 3))

    # MLP for final processing
    def mlp_block(x, output_size, use_bn=False):
        x = hk.Linear(output_size)(x)
        x = jax.nn.leaky_relu(x)
        if use_bn:
            x = hk.BatchNorm(True, True, 0.99)(x, is_training=is_training)
        return x#hk.dropout(hk.next_rng_key(), dropout_rate, is_training)(x)

    # Combine convolutional features with global data
    batch_global = mlp_block(batch_global, 512)
    combined_features = jnp.concatenate([batch_conv, batch_global], axis=-1)



    x = mlp_block(combined_features, 1024)
    x = mlp_block(x, 512)
    x = mlp_block(x, 256)
    x = mlp_block(x, 128)

    return hk.Linear(output_dim)(x)


# converts physical position to voxel position.

def r2a(pos, arr):
    return hp["conversionFactor"] * np.matmul(pos, arr)


def atom_to_array(position, axes):
    return r2a(position, axes) + hp["centre"] - r2a(np.array([1/2,1/2,1/2]), axes)

# converts voxel position to physical position.


def array_to_atom(position, axes, axes_inv):
    basePoint = hp["centre"] - r2a(np.array([1/2,1/2,1/2]), axes)
    return np.matmul((position - basePoint) / hp["conversionFactor"], axesInv)


def given_point_determine_cube_and_overlap(position):
    index = np.round(position).astype(int)

    def comp(i, p):
        return -1 if p < i else 1

    indices = list(
        itertools.product(*[[0, comp(index[i], position[i])] for i in range(3)])
    )
    points = [tuple(index + np.array(indices[i])) for i in range(8)]
    shared_vol = [np.prod(1 - np.abs(position - i)) for i in points]
    return (points, shared_vol)


def get_global_data_vector(poscar):
    return np.array(
        [unpack_line(poscar[2]), unpack_line(poscar[3]), unpack_line(poscar[4])]
    )


# uses mutability  @jax.jit


def test_convexity(index, axes, inv_axes):
    a = np.apply_along_axis(lambda r: array_to_atom(r, axes, inv_axes), 0, index)
    b = (0.0 < a) & (a < 1.0)
    c = np.all(b, axis=0)
    return c if 1 else 0


output_dim = 0  # initialize before calling the network


def unpack_data(index, data_item, ccnn_depth, random_rotate=False):

    # prep convnet
    dense_encode = np.zeros(
        (hp["maxDims"], hp["maxDims"], hp["maxDims"], ccnn_depth + 1)
    )

    axes = data_item[2]

    for i in range(len(data_item[0])):
        for j in complete_ternary:
            # this tiles out everything, then i dither the pixels
            points, vol = given_point_determine_cube_and_overlap(data_item[0][i] + r2a(np.array(j), axes))
            for jj in range(len(points)):  # use the variable encoding here.
                if (
                    -1 < points[jj][0] < hp["maxDims"]
                    and -1 < points[jj][1] < hp["maxDims"]
                    and -1 < points[jj][2] < hp["maxDims"]
                ):
                    dense_encode[points[jj][0], points[jj][1], points[jj][2], :] = [
                        vol[jj],
                        *(data_item[1][i]),
                    ]

    inv_axes = np.linalg.inv(axes)
    mask = np.fromfunction(
        lambda i, j, k, l: test_convexity(np.array((i, j, k)), axes, inv_axes),
        hp["dims"],
    )

    space = np.concatenate((dense_encode, mask), axis=-1)

    # this is the data that is passed to the network, needs concatenation with
    # global data still.
    return (index, space)


def preprocess_data(index):
    poscar = preprocess_poscar(index)

    # fixed dimensional
    pi_ = [poscar_global.info(poscar) for poscar_global in poscar_globals] + [
        global_input.info(index) for global_input in global_inputs
    ]

    # we do a partial implantation into the coercive structure in order to
    # determine which if any materials are issues.
    axes = get_global_data_vector(poscar)
    numbs = poscar[6].split()
    encoding = set()

    total = 0
    for ii in range(len(numbs)):
        total += int(numbs[ii])
        numbs[ii] = total

    # the atoms are encoded in two sets in the pa since this is a different
    # encoding type
    positions_ = [atom_to_array(unpack_line(poscar[8 + ii]), axes) for ii in range(total)]

    inputs_ = np.array(
        [
            sum(items, [])
            for items in zip(
                *[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics]
            )
        ]
    )
    pa_ = (positions_, inputs_, axes)  # the axes are for unpacking the mask

    # i don't currently check if the primitive cell is outside the
    # transformation - FIX
    for pos_ in positions_:
        if not (
            -1 < pos_[0] < hp["maxDims"]
            and -1 < pos_[1] < hp["maxDims"]
            and -1 < pos_[2] < hp["maxDims"]
        ):
            return ("too far", None, None, None)

        for j in completeTernary:
            # This tiles out everything, then I dither the pixels
            points, vol = givenPointDetermineCubeAndOverlap(
                pos_ + r2a(np.array(j), axes)
            )
            for jj in range(len(points)):
                if (
                    -1 < points[jj][0] < hp["maxDims"]
                    and -1 < points[jj][1] < hp["maxDims"]
                    and -1 < points[jj][2] < hp["maxDims"]
                ):
                    if points[jj] not in encoding:
                        encoding.add((points[jj][0], points[jj][1], points[jj][2]))
                    else:
                        print(index)
                        print("too close")
                        return ("too close", None, None, None)

    # fixed dimensional
    go_ = []
    for func in range(len(global_outputs)):
        go_.append(global_outputs[func].info(index))

    return ("good", flatten(pi_), pa_, flatten(go_))


if __name__ == "__main__":
    # The primary purpose of pickling here is to get rid of atomic structures
    # that are too large.
    if not exists("CCNN.pickle"):

        # the atomic embeddings are assumed to be both input and available for all
        # poscar files. The global embeddings are assumed to not exist for all elements.
        # The poscar_globals is just the cell size. The others should be
        # obvious.

        # Print the number of poscars total:
        setOfAllPoscars = getSetOfPoscars()
        print("There are", len(setOfAllPoscars), "poscars total.")

        ids = set.intersection(
            *flatten(
                [i.valid_ids() for i in global_inputs]
                + [i.valid_ids() for i in global_outputs]
            )
        )

        print("There are", len(ids), "poscars with all inputs and outputs.")

        # There are how many poscars missing:
        ids = set.intersection(ids, setOfAllPoscars)
        print(
            "There are",
            len(ids),
            "poscars with all inputs and outputs and all poscars.",
        )

        pi = []
        pa = []
        go = []
        go_types = [i.classifier() for i in global_outputs]
        go_nums = [len(i.info(str(1))) for i in global_outputs]

        atomsOutOfEmbedding = 0
        atomsTooClose = 0

        def accumulate_data(args):
            global atomsOutOfEmbedding
            global atomsTooClose
            global pi
            global pa
            global go

            if len(pi) % 1000 == 0:
                print(len(pi), "poscars processed")

            if args[0] == "good":
                pi.append(args[1])
                pa.append(args[2])
                go.append(args[3])
            elif args[0] == "too far":
                atomsOutOfEmbedding += 1
            elif args[0] == "too close":
                atomsTooClose += 1

        # Create a pool of worker processes
        pool = Pool(processes=hp["num_proc"])

        for index in list(ids):
            pool.apply_async(preprocess_data, args=(index,), callback=accumulate_data)

        pool.close()
        pool.join()
        print(len(pa))

        print(
            atomsOutOfEmbedding,
            "poscars were skipped due atoms being too far from the centre",
        )
        print(
            atomsTooClose,
            "poscars were skipped due to atoms being too close to each other",
        )

        with open("CCNN.pickle", "wb") as f:
            pickle.dump(
                {
                    "data_local": pa,  # This is unpacked on the other side
                    "data_global": np.array(pi),
                    "labels": np.array(go),
                    "labels_types": go_types,
                    "labels_nums": go_nums,
                },
                f,
            )

    with open("CCNN.pickle", "rb") as f:
        data = pickle.load(f)
        inputs_global = data["data_global"]
        inputs = data["data_local"]
        labels = data["labels"]
        label_types = data["labels_types"]
        label_nums = data["labels_nums"]

        output_dim = sum(label_nums)

        # Split the data into training and validation sets
        # X_train, y_train, add_train, X_val, y_val, add_val = partition_dataset(0.4, data, labels, additional_data)
        # The following data is still lightweight
        X_train, globals_train, y_train, X_val, globals_val, y_val = partition_dataset(
            0.1, inputs, inputs_global, labels
        )

        # This is the number of channels in the convolutional network.
        ccnn_depth = X_train[0][1].shape[-1]
        print("The ccnn depth is: ", ccnn_depth)
        # Initialize the network
        net = hk.transform_with_state(net_fn)
        rng = jax.random.PRNGKey(0x09F911029D74E35BD84156C5635688C0 % 2**32)
        init_rng, train_rng = jax.random.split(rng)
        params, state = net.init(
            init_rng,
            (
                jnp.array([unpack_data(0, X_train[0], ccnn_depth)[1]]),
                jnp.array([globals_train[0]]),
            ),
            is_training=True,
        )
        if exists("CCNN.params"):
            with open("CCNN.params", "rb") as f:
                params = pickle.load(f)
            print("Loaded model from CCNN.params")

        # Create the optimizer
        # Learning rate schedule: linear ramp-up and then constant
        num_epochs = 3
        num_batches = len(X_train) // hp["batch_size"]
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
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        try:
            # Now, for the unpacking, we need multithreading
            manager = Manager()
            results_queue = manager.Queue()
            training_acc = ExponentialDecayWeighting(0.99)
            training_res = []
            iii = 0

            # Create a pool of worker processes
            pool = Pool(processes=hp["num_proc"])

            for _ in range(num_epochs):
                for index, data_item in enumerate(inputs):
                    pool.apply_async(
                        unpack_data,
                        args=(index, data_item, ccnn_depth),
                        callback=results_queue.put,
                    )

            pool.apply_async(lambda a: None, args=(None,), callback=results_queue.put)

            batch = []

            while True:
                result = results_queue.get()

                if result is None:
                    # There will be less than a batch of data left, so just
                    # flush it.
                    break

                batch.append(result)

                if len(batch) == hp["batch_size"]:
                    conv_data = np.array([i[1] for i in batch])
                    g_data = np.array([inputs_global[i[0]] for i in batch])
                    label_data = np.array([labels[i[0]] for i in batch])

                    (loss, (accs, state)), grad = val_and_grad(params, state, rng, (conv_data, g_data), label_data)
                    updates, opt_state = opt_update(grad, opt_state)
                    params = optax.apply_updates(params, updates)

                    training_acc.add_accuracy(accs)

                    training_res.append(training_acc.get_weighted_average())

                    iii += 1
                    if iii % 10 == 0:#adjust per your feelings
                        print("updated graph", iii, "times")
                        line.set_xdata(np.arange(len(training_res)))
                        line.set_ydata(np.array(training_res))
                        ax.relim()
                        ax.autoscale_view()
                        fig.canvas.draw()
                        fig.canvas.flush_events()

                    batch.clear()

            pool.close()
            pool.join()

            # Save the training and validation loss
            # train_accuracy = compute_accuracy_fn(params, batch_rng, X_train, y_train)
            # val_accuracy = compute_accuracy_fn(params, batch_rng, X_val, y_val)
            # print(f"Epoch {epoch}, Training accuracy: {train_accuracy}, Validation accuracy: {val_accuracy}")
        except KeyboardInterrupt:
            with open("CCNN.params", "wb") as f:
                pickle.dump(params, f)
            print("Keyboard interrupt, saving model")

        with open("CCNN.params", "wb") as f:
            pickle.dump(params, f)
        
        plt.ioff()  # Turn off interactive mode
        plt.savefig('training_loss_CCNN.png')
        plt.close()
        print("Done training")
