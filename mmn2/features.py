import cv2
import numpy as np
import pandas as pd
import scipy.misc
from skimage import feature


def hog_representation(image, orientations, pixelsPerCell, cellsPerBlock, block_norm):
    hist = feature.hog(image, orientations=orientations,
                       pixels_per_cell=pixelsPerCell,
                       cells_per_block=cellsPerBlock,
                       block_norm=block_norm)
    return hist


def hog_batch_representation(images, orientations, pixelsPerCell, cellsPerBlock, block_norm):
    result = []
    for image in images:
        # describe the image and update the data matrix
        hist = hog_representation(image, orientations, pixelsPerCell, cellsPerBlock, block_norm)
        result.append(hist)
    return result


def sift_batch_representation(images, pickled_db_path="sift_features.pck"):
    result = []
    for index, image in images.iterrows():
        # print('Extracting features from image %s' % str(index))
        rgb = scipy.misc.toimage(image.values.reshape(28, 28))
        rgb = rgb.convert('RGB')
        open_cv_image = np.array(rgb)
        dsc = extract_sift_features(open_cv_image)
        result.append(dsc)
    # saving all our feature vectors in pickled file
    # with open(pickled_db_path, 'w') as fp:
    #     pickle.dump(result, fp)
    return result


def extract_sift_features(image, vector_size=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dsc = np.zeros(dtype='float32', shape=(1, 128))
    try:
        # Using KAZE, cause SIFT, ORB and other was moved to additional module
        # which is adding addtional pain during install
        alg = cv2.xfeatures2d.SIFT_create(edgeThreshold=20, contrastThreshold=0)
        # Dinding image keypoints
        kps = alg.detect(image)
        # Getting first 32 of them.
        # Number of keypoints is varies depend on image size and color pallet
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
        # computing descriptors vector
        if np.size(kps) > 0:
            kps, dsc = alg.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 128
        needed_size = (vector_size * 128)
        if dsc.size < needed_size:
            # if we have less the 10 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
    except cv2.error as e:
        print('Error: ', e)
        return None
    return dsc


if __name__ == '__main__':
    # load data- prepare sift features file
    # images = load_img.load_train_from_dir()
    # result = sift_batch_representation(images)

    mnist_dir = "MNIST\\"
    test = True
    test_size = 5000

    # 1. load data set
    train_data = pd.read_csv(mnist_dir + "train.csv")
    train_data.reset_index()

    np.random.seed(0)
    n_sample = len(train_data)
    order = np.random.permutation(n_sample)
    train_data = train_data.iloc[order]

    if test:
        train_data = train_data.head(test_size)

    X = train_data.drop("label", 1)
    y = train_data['label']

    digits = np.asarray(X).reshape((X.shape[0], 28, 28))  # remove head line
    hog_rep = hog_batch_representation(digits, orientations=3, pixelsPerCell=(2, 2),
                                       cellsPerBlock=(4, 4), block_norm='L2-Hys')
