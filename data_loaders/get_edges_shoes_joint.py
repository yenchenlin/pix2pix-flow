import numpy as np


def downsample(x, resolution):
    assert x.dtype == np.float32
    assert x.shape[1] % resolution == 0
    assert x.shape[2] % resolution == 0
    if x.shape[1] == x.shape[2] == resolution:
        return x
    s = x.shape
    x = np.reshape(x, [s[0], resolution, s[1] // resolution,
                       resolution, s[2] // resolution, s[3]])
    x = np.mean(x, (2, 4))
    return x


def x_to_uint8(x):
    x = np.clip(np.floor(x), 0, 255)
    return x.astype(np.uint8)


def shard(data, shards, rank):
    # Determinisitc shards
    x, y = data
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] % shards == 0
    assert 0 <= rank < shards
    size = x.shape[0] // shards
    ind = rank*size
    return x[ind:ind+size], y[ind:ind+size]


def get_data(problem, shards, rank, data_augmentation_level, n_batch_train, n_batch_test, n_batch_init, resolution, flip_color=False, code_path=None, shuffle_train=True):
    if problem == 'edges2shoes':
        x_train_A = np.load('/afs/csail.mit.edu/u/y/yenchenlin/Workspace/pix2pix/datasets/edges2shoes_downsample/train/shoes.npy')
        x_train_A = x_train_A[:-1, :, :, :] # To train on multiple GPUs, remove last training examples
        x_test_A = np.load('/afs/csail.mit.edu/u/y/yenchenlin/Workspace/pix2pix/datasets/edges2shoes_downsample/val/shoes.npy')

        x_train_B = np.load('/afs/csail.mit.edu/u/y/yenchenlin/Workspace/pix2pix/datasets/edges2shoes_downsample/train/edges.npy')
        x_train_B = x_train_B[:-1, :, :, :] # To train on multiple GPUs, remove last training examples
        x_test_B = np.load('/afs/csail.mit.edu/u/y/yenchenlin/Workspace/pix2pix/datasets/edges2shoes_downsample/val/edges.npy')

        y_train = np.zeros((x_train_A.shape[0]))
        y_test = np.zeros((x_test_A.shape[0]))
        y_train = np.reshape(y_train, [-1])
        y_test = np.reshape(y_test, [-1])
        y_train_A, y_train_B = y_train, y_train # Set up dummy y
        y_test_A, y_test_B = y_test, y_test # Set up dummy y
    else:
        raise Exception()


    print('n_train:', x_train_A.shape[0], 'n_test:', x_test_A.shape[0])

    # Shard before any shuffling
    x_train_A, y_train_A = shard((x_train_A, y_train_A), shards, rank)
    x_test_A, y_test_A = shard((x_test_A, y_test_A), shards, rank)

    x_train_B, y_train_B = shard((x_train_B, y_train_B), shards, rank)
    x_test_B, y_test_B = shard((x_test_B, y_test_B), shards, rank)

    print('n_shard_train:', x_train_A.shape[0], 'n_shard_test:', x_test_A.shape[0])

    from keras.preprocessing.image import ImageDataGenerator
    datagen_test = ImageDataGenerator()
    if data_augmentation_level == 0:
        datagen_train = ImageDataGenerator()
    else:
        if problem == 'edges2shoes':
            datagen_train = ImageDataGenerator(
                width_shift_range=0.1,
                height_shift_range=0.1
            )
        else:
            raise Exception()

    seed = 420
    datagen_train.fit(x_train_A, seed=seed)
    datagen_test.fit(x_test_A, seed=seed)
    train_flow_A = datagen_train.flow(x_train_A, y_train_A, n_batch_train, seed=seed)
    test_flow_A = datagen_test.flow(x_test_A, y_test_A, n_batch_test, shuffle=shuffle_train, seed=seed)

    datagen_train.fit(x_train_B, seed=seed)
    datagen_test.fit(x_test_B, seed=seed)
    train_flow_B = datagen_train.flow(x_train_B, y_train_B, n_batch_train, seed=seed)
    test_flow_B = datagen_test.flow(x_test_B, y_test_B, n_batch_test, shuffle=shuffle_train, seed=seed)

    def make_iterator(flow, resolution, code_path=None):
        def iterator():
            x_full, yz = flow.next()
            x_full = x_full.astype(np.float32)
            x = downsample(x_full, resolution)
            x = x_to_uint8(x)
            if code_path != None:
                y = np.squeeze(yz[:, :1])
                z = yz[:, 1:]
                return x, y, z
            else:
                y = yz
                return x, y

        return iterator

    #init_iterator = make_iterator(train_flow, resolution)
    train_iterator_A = make_iterator(train_flow_A, resolution, code_path)
    test_iterator_A = make_iterator(test_flow_A, resolution, code_path)

    train_iterator_B = make_iterator(train_flow_B, resolution, code_path)
    test_iterator_B = make_iterator(test_flow_B, resolution, code_path)

    # Get data for initialization
    data_init_A = make_batch(train_iterator_A, n_batch_train, n_batch_init, code_path=code_path)
    data_init_B = make_batch(train_iterator_B, n_batch_train, n_batch_init, code_path=code_path)

    return train_iterator_A, test_iterator_A, data_init_A, train_iterator_B, test_iterator_B, data_init_B


def make_batch(iterator, iterator_batch_size, required_batch_size, code_path=None):
    ib, rb = iterator_batch_size, required_batch_size
    #assert rb % ib == 0
    k = int(np.ceil(rb / ib))
    xs, ys, codes = [], [], []
    for i in range(k):
        if code_path != None:
            x, y, code = iterator()
            codes.append(code)
        else:
            x, y = iterator()
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs)[:rb], np.concatenate(ys)[:rb]
    if code_path != None:
        code = np.concatenate(codes)[:rb]
        return {'x': x, 'y': y, 'code': code}
    else:
        return {'x': x, 'y': y}
