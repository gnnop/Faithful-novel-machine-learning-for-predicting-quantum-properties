from struct import unpack
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
if __name__ == '__main__':
    copy_file_to_directories('../includes.py', target_directories)


#Now the rest of the file.

from includes import *
load_submodules() #this is a function in includes.py that loads all the submodules.


# hyperparameters
hp = {
    "dropoutRate": 0.1,
    "max_atoms": 40,
    "batch_size": 32,
    "maxDims": 48,#The input vector is a 3D lattice of size maxDims x maxDims x maxDims of atoms, represented as vectors of size maxRep.
    "conversionFactor": 1.45,#It is assumed that all atoms have the same size. This is the relative radius of each atom in lattice units.
    "num_proc": 8,
}

hp["centre"] = np.array( [hp["maxDims"] / 2, hp["maxDims"] / 2, hp["maxDims"] / 2] )
hp["dims"] = ( [hp["maxDims"], hp["maxDims"], hp["maxDims"]] ,-1)#The last entry is dynamically created before network operation



def net_fn(batch_input, is_training=False, dropout_rate=0):#Using the includes file loss functions, this is necessary for it.
  
  batch_conv, batch_global = batch_input

  #Literally insane. This is still too large
  cnet = hk.Sequential([
    hk.Conv3D(output_channels=40, kernel_shape=5, stride=2), jax.nn.leaky_relu,
    hk.Conv3D(output_channels=40, kernel_shape=5, stride=2), jax.nn.leaky_relu,
    hk.Conv3D(output_channels=40, kernel_shape=5, stride=2), jax.nn.leaky_relu,
    hk.Conv3D(output_channels=30, kernel_shape=3, stride=2), jax.nn.leaky_relu])
  
  y1 = cnet(batch_conv)

  #DANGER! The MLP conversion is the first number * y1.size. In a batch, this
  #easily exceeds single GPU VRAM with 62^3 * 4 * 20 * 20 * ? = 50 GB. I have 8GB, so reduce everything!

  mlp = hk.Sequential([
    hk.Flatten(),
    hk.Linear(800), jax.nn.leaky_relu,
    hk.Linear(600), jax.nn.leaky_relu,
    hk.Linear(200), jax.nn.leaky_relu,
    hk.Linear(60), jax.nn.leaky_relu,
    hk.Linear(40), jax.nn.leaky_relu,
    hk.Linear(20), jax.nn.leaky_relu,
    hk.Linear(output_dim)
  ])

  return mlp(jnp.concatenate((jnp.reshape(y1, (y1.shape[0], -1)), batch_global), axis=-1))

#Converts physical position to voxel position.
def atomToArray(position, axes):
    basePoint = hp["centre"] - hp["conversionFactor"] * ((axes[0] + axes[1] + axes[2]) / 2)
    return hp["conversionFactor"] * np.matmul(position, axes) + basePoint#Is this right?

#Converts voxel position to physical position.
def arrayToAtom(position, axes, axesInv):
    basePoint = hp["centre"] - hp["conversionFactor"] * ((axes[0] + axes[1] + axes[2]) / 2)
    return np.matmul((position - basePoint) / hp["conversionFactor"], axesInv)

def givenPointDetermineCubeAndOverlap(position):
    index = np.round(position).astype(int)
    comp = lambda i, p : -1 if p < i else 1
    indices = list(itertools.product(*[[0, comp(index[i], position[i])] for i in range(3)]))
    points = [tuple(index + np.array(indices[i])) for i in range(8)]
    shared_vol = [np.prod( 1 - np.abs(position - i)) for i in points]
    return (points, shared_vol)

def getGlobalDataVector(poscar):
    return np.array([unpackLine(poscar[2]), unpackLine(poscar[3]), unpackLine(poscar[4])])

#Uses mutability  @jax.jit
def testConvexity(index, axes, invAxes):
  a = np.apply_along_axis(lambda r: arrayToAtom(r, axes, invAxes), 0, index)
  b = (0.0 < a) & (a < 1.0)
  c = np.all(b, axis=0)
  return c if 1 else 0

output_dim = 0#initialize before calling the network

def unpack_data(index, data_item, ccnn_depth, randomRotate=False):

    #prep convnet
    denseEncode = np.zeros((hp["maxDims"], hp["maxDims"], hp["maxDims"],ccnn_depth + 1))

    for i in range(len(data_item[0])):
        for _ in completeTernary:
            #This tiles out everything, then I dither the pixels
            points, vol = givenPointDetermineCubeAndOverlap(data_item[0][i])
            for jj in range(len(points)):#Use the variable encoding here.
                if -1 < points[jj][0] < hp["maxDims"] and -1 < points[jj][1] < hp["maxDims"] and -1 < points[jj][2] < hp["maxDims"] and points[jj][0] >= 0 and points[jj][1] >= 0 and points[jj][2] >= 0:
                    denseEncode[points[jj][0], points[jj][1], points[jj][2], :] = [vol[jj], *(data_item[1][i])]
    
    axes = data_item[2]
    invAxes = np.linalg.inv(axes)
    mask = np.fromfunction(lambda i, j, k, l: testConvexity(np.array((i,j,k)), axes, invAxes), 
                    (hp["maxDims"], hp["maxDims"], hp["maxDims"], 1))

    space = np.concatenate((denseEncode, mask), axis=-1)

    return (index, space) #This is the data that is passed to the network, needs concatenation with global data still.


def preprocess_data(index):
    poscar = preprocessPoscar(index)

    #fixed dimensional
    pi_ = [poscar_global.info(poscar) for poscar_global in poscar_globals] + [global_input.info(index) for global_input in global_inputs]

    #We do a partial implantation into the coercive structure in order to determine which if any materials are issues.
    axes = getGlobalDataVector(poscar)
    numbs = poscar[6].split()
    encoding = set()

    total = 0
    for ii in range(len(numbs)):
        total+=int(numbs[ii])
        numbs[ii] = total
    
    #The atoms are encoded in two sets in the pa since this is a different encoding type
    positions_ = [atomToArray(unpackLine(poscar[8+ii]), axes) for ii in range(total)]

    inputs_ = np.array([sum(items, []) for items in zip(*[poscar_atomic.info(poscar) for poscar_atomic in poscar_atomics])])
    pa_ = (positions_, inputs_, axes)#The axes are for unpacking the mask

    #I don't currently check if the primitive cell is outside the transformation - FIX
    for pos_ in positions_:
        if not (-1 < pos_[0] < hp["maxDims"] and -1 < pos_[1] < hp["maxDims"] and -1 < pos_[2] < hp["maxDims"]):
            return ("too far", None, None, None)
        
        for j in completeTernary:
            #This tiles out everything, then I dither the pixels
            points, vol = givenPointDetermineCubeAndOverlap(atomToArray(pos_ + np.array(j), axes))
            for jj in range(len(points)):
                if -1 < points[jj][0] < hp["maxDims"] and -1 < points[jj][1] < hp["maxDims"] and -1 < points[jj][2] < hp["maxDims"]:
                    if points[jj] not in encoding:
                        encoding.add((points[jj][0], points[jj][1], points[jj][2]))
                    else:
                        return ("too close", None, None, None)

                
    

    #fixed dimensional
    go_ = []
    for func in range(len(global_outputs)):
        go_.append(global_outputs[func].info(index))

    return ("good", flatten(pi_), pa_, flatten(go_))


if __name__ == '__main__':
    #The primary purpose of pickling here is to get rid of atomic structures that are too large.
    if not exists("CCNN.pickle"):

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
            pool.apply_async( preprocess_data, args=(index,) , callback=accumulate_data )

        pool.close()
        pool.join()
        print(len(pa))

        print(atomsOutOfEmbedding, "poscars were skipped due atoms being too far from the centre")
        print(atomsTooClose, "poscars were skipped due to atoms being too close to each other")


        with open("CCNN.pickle", "wb") as f:
            pickle.dump({
                "data_local": pa,#This is unpacked on the other side
                "data_global": np.array(pi),
                "labels": np.array(go),
                "labels_types": go_types,
                "labels_nums": go_nums
            },f)


    with open("CCNN.pickle", "rb") as f:
        data = pickle.load(f)
        inputs_global = data["data_global"]
        inputs = data["data_local"]
        labels = data["labels"]
        label_types = data["labels_types"]
        label_nums = data["labels_nums"]

        output_dim = sum(label_nums)


        # Split the data into training and validation sets
        ##X_train, y_train, add_train, X_val, y_val, add_val = partition_dataset(0.4, data, labels, additional_data)
        #The following data is still lightweight
        X_train, globals_train, y_train, X_val, globals_val, y_val = partition_dataset(0.1, inputs, inputs_global, labels)


        ccnn_depth = X_train[0][1].shape[-1]#This is the number of channels in the convolutional network.
        print("The ccnn depth is: ", ccnn_depth)
        # Initialize the network
        net = hk.transform(net_fn)
        rng = jax.random.PRNGKey(0x09F911029D74E35BD84156C5635688C0 % 2**32)
        init_rng, train_rng = jax.random.split(rng)
        params = net.init(init_rng, (jnp.array([unpack_data(0, X_train[0], ccnn_depth)[1]]), jnp.array([globals_train[0]])), is_training=True)
        if exists("CCNN.params"):
            with open("CCNN.params", "rb") as f:
                params = pickle.load(f)
            print("Loaded model from CCNN.params")

        
        # Create the optimizer
        # Learning rate schedule: linear ramp-up and then constant
        num_epochs = 3
        num_batches = len(X_train) // hp["batch_size"]
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
            #Now, for the unpacking, we need multithreading
            manager = Manager()
            results_queue = manager.Queue()
            training_acc = ExponentialDecayWeighting(0.99)

            # Create a pool of worker processes
            pool = Pool(processes=hp["num_proc"])

            for _ in range(num_epochs):
                for index, data_item in enumerate(inputs):
                    pool.apply_async( unpack_data, args=(index, data_item,ccnn_depth) , callback=results_queue.put )
            
            pool.apply_async(lambda a: None, args=(None,), callback=results_queue.put)

            batch = []

            while True:
                result = results_queue.get()

                if result is None:
                    #There will be less than a batch of data left, so just flush it.
                    break

                batch.append(result)

                if len(batch) == hp["batch_size"]:
                    conv_data = np.array( [i[1] for i in batch] )
                    g_data = np.array( [inputs_global[i[0]] for i in batch] )
                    label_data = np.array( [labels[i[0]] for i in batch] )

                    (loss, accs), grad = jax.value_and_grad(compute_loss_fn, has_aux=True)(params, rng, (conv_data, g_data), label_data)
                    updates, opt_state = optimizer.update(grad, opt_state)
                    params = optax.apply_updates(params, updates)


                    training_acc.add_accuracy(accs)
                    print("trainging accuracy: ", training_acc.get_weighted_average())

                    batch.clear()
            
            pool.close()
            pool.join()


            # Save the training and validation loss
            #train_accuracy = compute_accuracy_fn(params, batch_rng, X_train, y_train)
            #val_accuracy = compute_accuracy_fn(params, batch_rng, X_val, y_val)
            #print(f"Epoch {epoch}, Training accuracy: {train_accuracy}, Validation accuracy: {val_accuracy}")
        except KeyboardInterrupt:
            with open("CCNN.params", "wb") as f:
                pickle.dump(params, f)
            print("Keyboard interrupt, saving model")
        
        with open("CCNN.params", "wb") as f:
            pickle.dump(params, f)
        print("Done training")
