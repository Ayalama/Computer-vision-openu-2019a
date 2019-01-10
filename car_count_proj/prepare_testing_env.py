import os
import shutil

# set test config- work with 20 test images and 1000 train images
IS_TEST = True
# read input images
DATA = 'input/'
IMAGES = 'input/Images/'
ANNOT = 'input/Annotations/'
if IS_TEST:
    DATA = 'test_input/'
    IMAGES = 'test_input/Images/'
    ANNOT = 'test_input/Annotations/'
TRAIN = os.path.join(DATA, 'ImageSets/train.txt')
TEST = os.path.join(DATA, 'ImageSets/test.txt')

with open(TRAIN, 'rt') as f: data = f.read().split('\n')[:-1]
TRAIN_IMG_LOC = [str('input/Images/' + line + '.png') for line in data]
TRAIN_IMG_ANN_LOC = [str('input/Annotations/' + line + '.txt') for line in data]

TRAIN_IMG_DST = [str(IMAGES + line + '.png') for line in data]
TRAIN_IMG_ANN_DEST = [str(ANNOT + line + '.txt') for line in data]

for i in range(len(TRAIN_IMG_LOC)):
    location = TRAIN_IMG_LOC[i]
    dest = TRAIN_IMG_DST[i]
    shutil.copy(location, dest)

    location = TRAIN_IMG_ANN_LOC[i]
    dest = TRAIN_IMG_ANN_DEST[i]
    shutil.copy(location, dest)

with open(TEST, 'rt') as f: data = f.read().split('\n')[:-1]
TEST_IMG_LOC = [str('input/Images/' + line + '.png') for line in data]
TEST_IMG_ANN_LOC = [str('input/Annotations/' + line + '.txt') for line in data]

TEST_IMG_DST = [str(IMAGES + line + '.png') for line in data]
TEST_IMG__ANN_DST = [str(ANNOT + line + '.txt') for line in data]

for i in range(len(TEST_IMG_LOC)):
    location = TEST_IMG_LOC[i]
    dest = TEST_IMG_DST[i]
    shutil.copy(location, dest)

    location = TEST_IMG_ANN_LOC[i]
    dest = TEST_IMG__ANN_DST[i]
    shutil.copy(location, dest)