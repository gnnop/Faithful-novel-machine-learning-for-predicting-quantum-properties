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

if __name__ == '__main__':
    jax.config.update("jax_enable_x64", False)

# hyperparameters
hp = {
    "dropoutRate": 0.1,
    "max_atoms": 40,
    "batch_size": 32,  # this is now baked into the preprocessing
    "num_proc": 8,  # this is the number of processes used in preprocessing
    "atom_dist": 4,#This is the distance under which atoms are connected by an edge.
}


@jraph.concatenated_args
def edge_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Edge update function for graph net."""
    net = hk.Sequential(
        [
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
        ]
    )
    return net(feats)


def edge_update_fn_skip(
    edges, sent_attributes, received_attributes, global_edge_attributes
):
    """Edge update function for graph net."""
    combined_args = jax.tree_util.tree_flatten(
        (edges, sent_attributes, received_attributes, global_edge_attributes)
    )[0]
    concat_args = jnp.concatenate(combined_args, axis=-1)
    net = hk.Sequential(
        [
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
        ]
    )
    return net(concat_args) + edges


@jraph.concatenated_args
def node_update_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Node update function for graph net."""
    net = hk.Sequential(
        [
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
        ]
    )
    return net(feats)


def node_update_fn_skip(nodes, sent_attributes, received_attributes, global_attributes):
    """Node update function for graph net."""
    combined_args = jax.tree_util.tree_flatten(
        (nodes, sent_attributes, received_attributes, global_attributes)
    )[0]
    concat_args = jnp.concatenate(combined_args, axis=-1)
    net = hk.Sequential(
        [
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
        ]
    )
    return net(concat_args) + nodes


@jraph.concatenated_args
def update_global_fn(feats: jnp.ndarray) -> jnp.ndarray:
    """Global update function for graph net."""
    # MUTAG is a binary classification task, so output pos neg logits.
    net = hk.Sequential(
        [
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
        ]
    )
    return net(feats)


def update_global_fn_skip(node_attributes, edge_attribtutes, globals_):
    """Global update function for graph net."""
    # MUTAG is a binary classification task, so output pos neg logits.
    combined_args = jax.tree_util.tree_flatten(
        (node_attributes, edge_attribtutes, globals_)
    )[0]
    concat_args = jnp.concatenate(combined_args, axis=-1)
    net = hk.Sequential(
        [
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
            hk.Linear(256),
            jax.nn.leaky_relu,
        ]
    )
    return net(concat_args) + globals_


def net_fn(graph, is_training=False, dropout_rate=0):
    global globals

    # The shape of the graph is wrong.
    # The graph globals structure relies on the globals being in
    # a matrix. Uhhhh

    collector = jraph.GraphMapFeatures(
        hk.Sequential([hk.Linear(256)]),
        hk.Sequential([hk.Linear(256)]),
        hk.Sequential(
            [
                hk.Linear(512),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(128),
                jax.nn.leaky_relu,
                hk.Linear(output_dim),
            ]
        ),
    )

    embedder = jraph.GraphMapFeatures(
        hk.Sequential(
            [
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
            ]
        ),
        hk.Sequential(
            [
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
            ]
        ),
        hk.Sequential(
            [
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
                hk.Linear(256),
                jax.nn.leaky_relu,
            ]
        ),
    )
    net1 = jraph.GraphNetwork(
        update_node_fn=node_update_fn_skip,
        update_edge_fn=edge_update_fn_skip,
        update_global_fn=update_global_fn_skip,
    )

    net2 = jraph.GraphNetwork(
        update_node_fn=node_update_fn_skip,
        update_edge_fn=edge_update_fn_skip,
        update_global_fn=update_global_fn_skip,
    )

    x1 = embedder(graph)
    x2 = net2(net1(x1))
    x3 = collector(x2)

    # return graph minus the fake one
    return x3.globals[0:-1]


output_dim = 0  # initialize before calling the network


def reflect_atom_dist(lis):
    return map(lambda a: [a[0], -a[1], -a[2], -a[3]], lis)


def atom_distance(axes, loc1, loc2):
    dist = []

    for i in complete_ternary:
        dir = np.array(loc1) - np.array(loc2) + np.matmul(i, axes)

        if np.linalg.norm(dir) < hp["atom_dist"]:
            dist.append([np.linalg.norm(dir), *dir])

    # do i need the subtraction info?
    return dist


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y


#graph creation things ALWAYS only appends one graph
#This is important for the masking operation.
def pad_graph_to_nearest_power_of_two(
    graphs_tuple: jraph.GraphsTuple,
) -> jraph.GraphsTuple:
    """Pads a batched `GraphsTuple` to the nearest power of two"""
    # add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_node)) + 1
    pad_edges_to = _nearest_bigger_power_of_two(jnp.sum(graphs_tuple.n_edge))
    # add 1 since we need at least one padding graph for pad_with_graphs.
    # we do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = graphs_tuple.n_node.shape[0] + 1
    return jraph.pad_with_graphs(
        graphs_tuple, pad_nodes_to, pad_edges_to, pad_graphs_to
    )


def extract_graph(atomic_encoding, absolute_position, glob, label_size, axes):
    global_data = [0.0] * (len(glob) + label_size)
    global_data[0 : len(glob)] = glob

    nodes_array = atomic_encoding

    # now we do an n^2 operation on the atoms:

    sender_array = []
    receiver_array = []
    edge_features = []

    for i in range(len(absolute_position)):
        for j in range(i+1):
            ls = atom_distance(axes, absolute_position[i], absolute_position[j])
            # this is a list of
            sender_array.extend([i] * len(ls))
            receiver_array.extend([j] * len(ls))
            edge_features.extend(ls)
            if i != j:
                sender_array.extend([j] * len(ls))
                receiver_array.extend([i] * len(ls))
                edge_features.extend(reflect_atom_dist(ls))

    # everything should be lined up now, but we need to add it to a graph

    return jraph.GraphsTuple(
        nodes=jnp.array(nodes_array),
        senders=jnp.array(sender_array),
        receivers=jnp.array(receiver_array),
        edges=jnp.array(edge_features),
        globals=jnp.array([global_data]),
        n_node=jnp.array([len(nodes_array)]),
        n_edge=jnp.array([len(sender_array)]),
    )


def get_global_data_vector(poscar):
    return np.array(
        [unpack_line(poscar[2]), unpack_line(poscar[3]), unpack_line(poscar[4])]
    )


def process_graph_batch(graph):
    poscar = preprocess_poscar(graph)
    numbs = poscar[6].split()

    total = 0
    for ii in range(len(numbs)):
        total += int(numbs[ii])
        numbs[ii] = total

    # fixed dimensional
    axes_ = get_global_data_vector(poscar)
    pi_ = flatten(
        [poscar_global.info(poscar) for poscar_global in poscar_globals]
        + [global_input.info(graph) for global_input in global_inputs]
    )
    inputs_ = np.array(
        [
            sum(items, [])
            for items in zip(
                *[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics]
            )
        ]
    )
    positions_ = [np.matmul(unpack_line(poscar[8 + ii]), axes_) for ii in range(total)]

    # fixed dimensional
    go_ = []
    for func in range(len(global_outputs)):
        go_.append(global_outputs[func].info(graph))
    go_ = sum(go_, [])

    graph_ = extract_graph(inputs_, positions_, pi_, len(go_), axes_)

    return graph_, go_


if __name__ == "__main__":
    if not exists("CGNN.pickle"):

        # the atomic embeddings are assumed to be both input and available for all
        # poscar files. the global embeddings are assumed to not exist for all elements.
        # the poscar_globals is just the cell size. the others should be
        # obvious.

        # print the number of poscars total:
        set_of_all_poscars = get_set_of_poscars()
        print("There are", len(set_of_all_poscars), "poscars total.")

        ids = set.intersection(
            *flatten(
                [i.valid_ids() for i in global_inputs]
                + [i.valid_ids() for i in global_outputs]
            )
        )

        print("There are", len(ids), "poscars with all inputs and outputs.")

        # there are how many poscars missing:
        ids = set.intersection(ids, set_of_all_poscars)
        print(
            "There are",
            len(ids),
            "poscars with all inputs and outputs and all poscars.",
        )

        data = []
        labels = []
        go_types = [i.classifier() for i in global_outputs]
        go_nums = [len(i.info(str(1))) for i in global_outputs]

        data_batch = []
        label_batch = []

        def accumulate_data(args):
            global data
            global labels
            global data_batch
            global label_batch

            data_batch.append(args[0])
            label_batch.append(args[1])

            if len(data_batch) == hp["batch_size"]:
                data.append(pad_graph_to_nearest_power_of_two(jraph.batch_np(data_batch)))
                labels.append(np.array(label_batch))
                data_batch = []
                label_batch = []
                print(len(data), "batches processed")

        pool = Pool(processes=hp["num_proc"])
        # was going to batch the graphs, but padding them is incredibly memory
        # intensive

        for index in list(ids):
            pool.apply_async(process_graph_batch, args=(index,), callback=accumulate_data)

        pool.close()
        pool.join()

        with open("CGNN.pickle", "wb") as f:
            pickle.dump(
                {
                    "data": data,
                    "labels": labels,
                    "labels_types": go_types,
                    "labels_nums": go_nums,
                },
                f,
            )

    with open("CGNN.pickle", "rb") as f:
        data = pickle.load(f)
        inputs = data["data"]
        labels = data["labels"]
        label_types = data["labels_types"]
        label_nums = data["labels_nums"]

        output_dim = sum(label_nums)

        # split the data into training and validation sets
        # x_train, y_train, add_train, x_val, y_val, add_val = partition_dataset(0.4, data, labels, additional_data)
        x_train, y_train, x_val, y_val = partition_dataset(0.1, inputs, labels)

        # now, attempt to load the model cgnn.params if it exists, otherwise init with haiku
        # initialize the network
        net = hk.transform_with_state(net_fn)
        rng = jax.random.PRNGKey(0x09_f911029_d74_e35_bd84156_c5635688_c0 % 2**32)
        init_rng, train_rng = jax.random.split(rng)
        params, state = net.init(init_rng, x_train[0], is_training=True)
        if exists("CGNN.params"):
            with open("CGNN.params", "rb") as f:
                params = pickle.load(f)
            print("Loaded model from CGNN.params")

        # create the optimizer
        # learning rate schedule: linear ramp-up and then constant
        num_epochs = 1000
        num_batches = len(x_train)
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
            training_acc = ExponentialDecayWeighting(0.99)
            training_res = []
            iii = 0

            for epoch in range(num_epochs):
                train_rng = jax.random.split(train_rng, num=1)[0]
                shuffled_indices = jax.random.permutation(train_rng, len(x_train))

                for i in range(num_batches):
                    batch_rng = jax.random.fold_in(train_rng, i)

                    (loss, (accs, state)), grad = val_and_grad(params, state, rng, x_train[shuffled_indices[i]], y_train[shuffled_indices[i]])
                    updates, opt_state = opt_update(grad, opt_state, params)
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

                
                batch_accuracies = []
                for batch_idx in range(len(y_val)):
                    batch_accuracy = compute_accuracy_fn(params,state, rng, x_val[batch_idx], y_val[batch_idx])
                    batch_accuracies.append(batch_accuracy)
                
                # Stack the batch accuracies and take the mean
                batch_accuracies = jnp.stack(batch_accuracies)
                mean_accuracy = jnp.mean(batch_accuracies, axis=0)

                print(f"Epoch {epoch}, Validation accuracy: {mean_accuracy}")
                # save the training and validation loss - just validation, not
                # implemented yet
        except KeyboardInterrupt:
            with open("CGNN.params", "wb") as f:
                pickle.dump(params, f)
            print("Keyboard interrupt, saving model")

        with open("CGNN.params", "wb") as f:
            pickle.dump(params, f)
        plt.ioff()  # Turn off interactive mode
        plt.savefig('training_loss_CGNN.png')
        plt.close()
        print("Done training")
