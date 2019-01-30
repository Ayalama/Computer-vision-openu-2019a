import itertools
import os
from random import shuffle
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import Sequential
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import *
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
from skimage.transform import resize
from sklearn.model_selection import train_test_split


print(os.listdir("../whales_task/input"))

DATA = "../whales_task/input"
TRAIN_IMG = "../whales_task/input/train"
TEST_IMG = "../whales_task/input/test"
BRANCH_MOD = "../input/simple-cnn-classification/xception_model_finetune.h5"
SUBMISSION_Df = '../whales_task/input/sample_submission.csv'
IM_SIZE = 299

test_df = pd.DataFrame({"Image": os.listdir(TEST_IMG)})
print("test images:" + str(len(test_df)))

train_lbl = pd.read_csv(os.path.join(DATA, 'train.csv'))
SUB_Df = pd.read_csv(os.path.join(DATA, 'sample_submission.csv'))
print("train images:" + str(len(train_lbl)))
print("total unique class:" + str(len(np.unique(train_lbl['Id']))))

###take out whales with a single example
df = train_lbl.groupby(['Id']).size().reset_index(
    name='train_examples')
df = df[df['train_examples'] < 2]
single_whale_set = set(df.Id.values)
print("number of classes with less than 2 examples:" + str(len(df)))


###triplets
def get_random_image(img_groups, group_names, gid):
    gname = group_names[gid]
    photos = img_groups[gname]
    pid = np.random.choice(np.arange(len(photos)), size=1)[0]
    pname = photos[pid]
    return pname


def create_triples(image_dir, labels, data_set='train'):
    img_groups = {}
    for img_file in os.listdir(image_dir):
        pid = img_file
        gid = labels[labels['Image'] == pid].values[0][1]  # this is the ralevant whale group
        if gid == 'new_whale': continue
        if gid in single_whale_set: continue
        #         if gid is None: continue
        if gid in img_groups:
            img_groups[gid].append(pid)
        else:
            img_groups[gid] = [pid]
    pos_triples, neg_triples = [], []
    # positive pairs are any combination of images in same group
    for key in img_groups.keys():
        #         triples = [(key + x[0] , key + x[1] , 1)
        #                  for x in itertools.combinations(img_groups[key], 2)]
        triples = [(x[0], x[1], 1)
                   for x in itertools.combinations(img_groups[key], 2)]
        pos_triples.extend(triples)
    # need equal number of negative examples
    group_names = list(img_groups.keys())
    for i in range(len(pos_triples)):
        g1, g2 = np.random.choice(np.arange(len(group_names)), size=2, replace=False)
        left = get_random_image(img_groups, group_names, g1)
        right = get_random_image(img_groups, group_names, g2)
        neg_triples.append((left, right, 0))
    pos_triples.extend(neg_triples)
    shuffle(pos_triples)
    return pos_triples


triples_data = create_triples(TRAIN_IMG, train_lbl)

print(len(triples_data))
print("triplets examples:")
print(triples_data[0:5])

#######data generator
RESIZE_IMG = IM_SIZE


def cached_imread(image_path, image_cache):
    if not image_path in image_cache:
        image = plt.imread(image_path).astype(np.float32)
        #         print(image.shape)
        image = resize(image, (RESIZE_IMG, RESIZE_IMG, 3))
        image_cache[image_path] = image
    return image_cache[image_path]


def preprocess_images(image_names, seed, datagen, image_cache):
    np.random.seed(seed)
    X = np.zeros((len(image_names), RESIZE_IMG, RESIZE_IMG, 3))
    for i, image_name in enumerate(image_names):
        #         print(image_name)
        if os.path.isfile(image_name):
            image = cached_imread(image_name, image_cache)
        else:
            image = cached_imread(os.path.join(TRAIN_IMG, image_name), image_cache)
        if datagen is not None:
            X[i] = datagen.random_transform(image)
        else:
            X[i] = image
    return X


def image_triple_generator(image_triples, batch_size):
    datagen_args = dict(rescale=1./255,rotation_range=10, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,
                        zoom_range=0.2, horizontal_flip=True)
    datagen_left = ImageDataGenerator(**datagen_args)
    datagen_right = ImageDataGenerator(**datagen_args)
    image_cache = {}
    while True:
        # loop once per epoch
        num_recs = len(image_triples)
        indices = np.random.permutation(np.arange(num_recs))
        num_batches = num_recs // batch_size
        for bid in range(num_batches):
            # loop once per batch
            batch_indices = indices[bid * batch_size: (bid + 1) * batch_size]
            batch = [image_triples[i] for i in batch_indices]
            # make sure image data generators generate same transformations
            seed = np.random.randint(low=0, high=1000, size=1)[0]
            Xleft = preprocess_images([b[0] for b in batch], seed,
                                      datagen_left, image_cache)
            Xright = preprocess_images([b[1] for b in batch], seed,
                                       datagen_right, image_cache)
            Y = np.array([b[2] for b in batch])  # 0 or 1
            #             Y = np_utils.to_categorical(np.array([b[2] for b in batch])) # 0 or 1
            yield ([Xleft, Xright], Y)


triples_batch_gen = image_triple_generator(triples_data, 32)
([Xleft, Xright], Y) = triples_batch_gen.__next__()
print("generator output shapes:")
print(Xleft.shape, Xright.shape, Y.shape)


######load model
def create_base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    # CONV => RELU => POOL
    seq.add(Conv2D(20, kernel_size=5, padding="same", input_shape=input_shape, activation='sigmoid'))
    seq.add(BatchNormalization())
    #     seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    seq.add(BatchNormalization())
    # CONV => RELU => POOL
    seq.add(Conv2D(50, kernel_size=5, padding="same", activation='sigmoid'))
    seq.add(BatchNormalization())
    #     seq.add(Activation("relu"))
    seq.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    seq.add(BatchNormalization())
    # Flatten => RELU
    seq.add(Flatten())
    seq.add(Dense(500))
    return seq


# Without Xception:
image_size = IM_SIZE
input_shape = (image_size, image_size, 3)
base_network = create_base_network(input_shape)
input_shape = (image_size, image_size, 3)
vector_left = Input(shape=base_network.output_shape[1:])
vector_right = Input(shape=base_network.output_shape[1:])
img_l = Input(shape=input_shape)
img_r = Input(shape=input_shape)
x_l = base_network(img_l)
x_r = base_network(img_r)

############head model
# layer to merge two encoded inputs with the l1 distance between them
mid        = 32
L_prod = Lambda(lambda x : x[0]*x[1])([vector_left, vector_right])
L_sum = Lambda(lambda x : x[0] + x[1])([vector_left, vector_right])
L1_distance= Lambda(lambda x : K.abs(x[0] - x[1]))([vector_left, vector_right])
L2_distance= Lambda(lambda x : K.square(x[0] - x[1]))([vector_left, vector_right])
distance= Concatenate()([L_prod, L_sum, L1_distance, L2_distance])
distance= Reshape((4, base_network.output_shape[1], 1), name='reshape1')(distance)
x = Conv2D(mid, (4, 1), activation='relu', padding='valid')(distance)
x = BatchNormalization()(x)
x = Reshape((base_network.output_shape[1], mid, 1))(x)
x = Conv2D(1, (1, mid), activation='linear', padding='valid')(x)
x = BatchNormalization()(x)
x = Flatten(name='flatten')(x)
pred = Dense(1, use_bias=True, activation='sigmoid', name='weighted-average',kernel_initializer="random_normal")(x)
head_model = Model([vector_left, vector_right], outputs=pred, name='head')

x = head_model([x_l, x_r])
siamese_model = Model(inputs=[img_l, img_r], outputs= x)

# try:
#     siamese_model = multi_gpu_model(siamese_model, gpus=3)
# except:
#     pass
siamese_model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])

############Train
triples_train, triples_test = train_test_split(triples_data, test_size=0.1, random_state=42)

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', patience=5, min_lr=1e-9, verbose=1, mode='min'),
    ModelCheckpoint('siamese_mid.h5', monitor='val_loss', save_best_only=True, verbose=1)
]

BATCH_SIZE = 32
NUM_EPOCHS = 100
image_cache = {}

train_gen = image_triple_generator(triples_train, BATCH_SIZE)
val_gen = image_triple_generator(triples_test, BATCH_SIZE)

# num_train_steps = len(triples_train) // BATCH_SIZE
# num_val_steps = len(triples_test) // BATCH_SIZE
num_train_steps = 100
num_val_steps = 40

history = siamese_model.fit_generator(train_gen,
                                      steps_per_epoch=num_train_steps,
                                      epochs=NUM_EPOCHS,
                                      validation_data=val_gen,
                                      validation_steps=num_val_steps,
                                      callbacks=callbacks)

siamese_model.save('siamese_trained.h5')
