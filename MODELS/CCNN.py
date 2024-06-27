from struct import unpack
import sys, os, shutil

from jraph import batch

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
    "maxDims": 48,#The input vector is a 3D lattice of size maxDims x maxDims x maxDims of atoms, represented as vectors of size maxRep.
    "conversionFactor": 1.45,#It is assumed that all atoms have the same size. This is the relative radius of each atom in lattice units.
    "num_proc": 12,
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
    hk.Linear(globals["labelSize"])
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
    #absolutely need this to not rearrange the atoms (it doesn't)
    inputs = np.array([poscar_atomic.info(data_item) for poscar_atomic in poscar_atomics])
    print(inputs.shape)

    #prep convnet
    denseEncode = np.zeros((hp["maxDims"], hp["maxDims"], hp["maxDims"],ccnn_depth + 1))
    print(denseEncode.shape)

    axes = getGlobalDataVector(data_item)
    if randomRotate:#This is currently unsafe, the rotation could orient the atomic structure outside the cube
        rot = R.random().as_matrix()#This is a random rotation matrix
        axes = np.matmul(rot, axes)
    
    atoms = data_item[5].split()
    numbs = data_item[6].split()

    total = 0
    for i in range(len(numbs)):
        total+=int(numbs[i])
        numbs[i] = total
    
    curIndx = 0
    atomType = 0
    for i in range(total):
        curIndx += 1
        if curIndx > numbs[atomType]:
            atomType += 1
        
        for j in completeTernary:
            #This tiles out everything, then I dither the pixels
            points, vol = givenPointDetermineCubeAndOverlap(atomToArray(unpackLine(data_item[8+i]) + np.array(j), axes))
            for jj in range(len(points)):#Use the variable encoding here.
                denseEncode[points[jj][0], points[jj][1], points[jj][2], :] = [vol[jj], *flatten(inputs[i])]
    
    invAxes = np.linalg.inv(axes)
    mask = np.fromfunction(lambda i, j, k, l: testConvexity(np.array((i,j,k)), axes, invAxes), 
                    (hp["maxDims"], hp["maxDims"], hp["maxDims"], 1))

    space = np.concatenate((denseEncode, mask), axis=-1)

    return (index, space) #This is the data that is passed to the network, needs concatenation with global data still.










#The primary purpose of pickling here is to get rid of atomic structures that are too large.
if not exists("processed.pickle"):

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
    go_nums = [len(i.info(1)) for i in global_outputs]

    atomsOutOfEmbedding = 0
    atomsTooClose = 0
    for i in ids:

        poscar = preprocessPoscar(i)

        #fixed dimensional
        pi_ = [poscar_global.info(poscar) for poscar_global in poscar_globals] + [global_input.info(i) for global_input in global_inputs]

        #We do a partial implantation into the coercive structure in order to determine which if any materials are issues.
        axes = getGlobalDataVector(poscar)
        numbs = poscar[6].split()
        encoding = set()

        total = 0
        for i in range(len(numbs)):
            total+=int(numbs[i])
            numbs[i] = total
        
        for i in range(total):
            actPos, _ = givenPointDetermineCubeAndOverlap(atomToArray(unpackLine(poscar[8+i]), axes))
            atomsOutOfEmbedding += 1
            continue

        for j in completeTernary:
            #This tiles out everything, then I dither the pixels
            points, vol = givenPointDetermineCubeAndOverlap(atomToArray(unpackLine(poscar[8+i]) + np.array(j), axes))
            for jj in range(len(points)):
                if -1 < points[jj][0] < hp["maxDims"] and -1 < points[jj][1] < hp["maxDims"] and -1 < points[jj][2] < hp["maxDims"] and points[jj][0] >= 0 and points[jj][1] >= 0 and points[jj][2] >= 0:
                    if points[jj] not in encoding:
                        encoding.add((points[jj][0], points[jj][1], points[jj][2]))
                    else:
                        print("Enlarge the encoding, this data is corrupted!")
                        print("The offending poscar is", poscar)
                        print(encoding)
                        print(points[jj])

                        exit()#hard failure

                        #if you're ok with losing some more atoms.
                        atomsTooClose += 1
                        continue
        

        #fixed dimensional
        go_ = []
        for func in range(len(global_outputs)):
            go_.append(global_outputs[func].info(i))


        pi.append(flatten(pi_))
        pa.append(poscar)
        go.append(flatten(go_))

    print(atomsOutOfEmbedding, "poscars were skipped due atoms being too far from the centre")
    print(atomsTooClose, "poscars were skipped due to atoms being too close to each other")


    with open("processed.pickle", "wb") as f:
        pickle.dump({
            "data_local": pa,#This is unpacked on the other side
            "data_global": np.array(pi),
            "labels": np.array(go),
            "labels_types": go_types,
            "labels_nums": go_nums
        },f)


with open("processed.pickle", "rb") as f:
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

    #Now, attempt to load the model CCNN.params if it exists, otherwise init with haiku

    print(X_train[0])
    print(poscar_atomics)
    print([poscar_atomic.info(X_train[0]) for poscar_atomic in poscar_atomics])
    ccnn_depth = sum([len(poscar_atomic.info(X_train[0])[0]) for poscar_atomic in poscar_atomics])
    print("The ccnn depth is: ", ccnn_depth)
    # Initialize the network
    net = hk.transform(net_fn)
    rng = jax.random.PRNGKey(0x09F911029D74E35BD84156C5635688C0 % 2**32)
    init_rng, train_rng = jax.random.split(rng)
    params = net.init(init_rng, (unpack_data(0, X_train[0], ccnn_depth)[1], globals_train[0]), is_training=True)
    if exists("CCNN.params"):
        with open("CCNN.params", "rb") as f:
            params = pickle.load(f)
        print("Loaded model from CCNN.params")

    
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
        #Now, for the unpacking, we need multithreading
        manager = Manager()
        results_queue = manager.Queue()

        # Create a pool of worker processes
        num_proc = 12
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
                g_data = np.array( [inputs_global[i[0]] for i in conv_data] )
                label_data = np.array( [labels[i[0]] for i in conv_data] )

                (loss, accs), grad = jax.value_and_grad(compute_loss_fn, has_aux=True)(params, rng, (conv_data, g_data), label_data)
                updates, opt_state = optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, updates)

                print(loss, accs)
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
