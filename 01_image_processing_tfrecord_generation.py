# Importing required libraries
import math, os, re, warnings, random, time
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import optimizers, losses, metrics, Model
from sklearn.manifold import TSNE
from tensorflow.keras import mixed_precision
import re, math, os, cv2, random, warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import glob, torch, imagehash
from tqdm.auto import tqdm
from PIL import Image

# Function to ensure consistent random seeds for reproducibility
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 0
seed_everything(seed)
warnings.filterwarnings('ignore')



# Setup TPU or default strategy for distributed computing
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f'Running on TPU {tpu.master()}')
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()




# Get the number of replicas (used for parallelism)
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')




# Enable mixed precision for better performance on supported hardware
policy = mixed_precision.Policy('mixed_bfloat16')
mixed_precision.set_global_policy(policy)

# Enable XLA (Accelerated Linear Algebra) for performance optimization
tf.config.optimizer.set_jit(True)




# Path to the directory containing subfolders with image files of a dataset, each folder represent one specific class.
# The path points to a directory containing images categorized into subfolders.
val_dir = '/g/data/nk53/mr3328/bracs/mhist/train/'

# Collect image filenames and their respective labels from directory structure
image_files = []
labels = []
for root, dirs, files in os.walk(val_dir):
    for file in files:
        if file.endswith('.png'):
            image_files.append(file)
            labels.append(os.path.basename(root))

# Create a DataFrame that stores image filenames and their corresponding labels
df = pd.DataFrame({'image_id': image_files, 'label': labels})

# Save the DataFrame as a CSV for future reference
output_path = 'train.csv'
df.to_csv(output_path, index=False)

# Copy all image files to a new directory for consistent file access during training
new_dir = '/g/data/nk53/mr3328/bracs/mhist/train_images/'
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

# Copy image files from original directory to the new directory
for root, dirs, files in os.walk(val_dir):
    for file in files:
        if file.endswith('.png'):
            src_path = os.path.join(root, file)
            dst_path = os.path.join(new_dir, file)
            shutil.copy(src_path, dst_path)

# Using image hashing techniques to detect duplicates in the dataset
# The images are now located in the `train_images/` directory, and we use hashing techniques to identify duplicates.
IMAGES_DIR = '/g/data/nk53/mr3328/bracs/mhist/train_images/'

# Define hash functions to identify duplicates
funcs = [imagehash.average_hash, imagehash.phash, imagehash.dhash, imagehash.whash]
image_ids = []
hashes = []

# Calculate image hashes and store them for each image
for path in tqdm(glob.glob(IMAGES_DIR + '*.png')):
    image = Image.open(path)
    image_id = os.path.basename(path)
    image_ids.append(image_id)
    hashes.append(np.array([f(image).hash for f in funcs]).reshape(256))

# Convert the hashes to a tensor for similarity comparison
hashes_all = np.array(hashes)
hashes_all = torch.Tensor(hashes_all.astype(int))

# Function to decode images from PNG format and resize them for the model
def decode_image(image_data):
    image = tf.image.decode_png(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [HEIGHT, WIDTH])
    image = tf.reshape(image, [HEIGHT, WIDTH, 3])
    return image

# Function to read a single TFRecord entry
def read_tfrecord(example):
    TFREC_FORMAT = {
        'image': tf.io.FixedLenFeature([], tf.string), 
        'target': tf.io.FixedLenFeature([], tf.int64), 
        'image_name': tf.io.FixedLenFeature([], tf.string), 
    }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    image = decode_image(example['image'])
    target = example['target']
    name = example['image_name']
    return image, target, name

# Function to load a dataset from TFRecord files
def load_dataset(filenames, HEIGHT, WIDTH, CHANNELS=3):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset

# Save a sample of images from the dataset for validation purposes
def save_samples(ds, row, col, output_dir='output'):
    ds_iter = iter(ds)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for j in range(row*col):
        image, label, name = next(ds_iter)
        filename = f"{label[0]}_{name[0].numpy().decode('utf-8')}.png"
        output_path = os.path.join(output_dir, filename)
        img = tf.cast(image[0], tf.uint8)
        img_array = img.numpy()
        plt.imsave(output_path, img_array)
        
    print(f"Saved {row*col} images in the '{output_dir}' directory.")

# Function to count the number of data samples in TFRecord files
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

# Helper functions to serialize features for TFRecord creation
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# Function to serialize images, labels, and filenames into TFRecord format
def serialize_example(image, target, image_name):
    feature = {
        'image': _bytes_feature(image),
        'target': _int64_feature(target),
        'image_name': _bytes_feature(image_name),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()



# Paths and configuration for processing images
# Set the number of splits for cross-validation (N_FILES), image size (HEIGHT, WIDTH), and image quality.
database_base_path = '/g/data/nk53/mr3328/bracs/mhist/'
PATH = f'{database_base_path}train_images/'
IMGS = os.listdir(PATH)
N_FILES = 5
HEIGHT, WIDTH = (512, 512)
IMG_QUALITY = 100
print(f'Image samples: {len(IMGS)}')


# Identifying duplicate images based on hash similarity
# The code below compares image hashes to identify duplicate images.
# It calculates the similarity between images by comparing their hash values.
# If the similarity is higher than 90%, the image is flagged as a duplicate.

sims = np.array([(hashes_all[i] == hashes_all).sum(dim=1).numpy()/256 for i in range(hashes_all.shape[0])])
indices1 = np.where(sims > 0.9)
indices2 = np.where(indices1[0] != indices1[1])
image_ids1 = [image_ids[i] for i in indices1[0][indices2]]
image_ids2 = [image_ids[i] for i in indices1[1][indices2]]
dups = {tuple(sorted([image_id1,image_id2])):True for image_id1, image_id2 in zip(image_ids1, image_ids2)}
duplicate_image_ids = sorted(list(dups))
print('found %d duplicates' % len(duplicate_image_ids))

# Removing duplicates from the training data
# Load the training CSV and remove rows corresponding to duplicate images.
imgs_to_remove = [x[1] for x in duplicate_image_ids]
remove_pd = []
for image in imgs_to_remove:
    remove_pd.append(image)
train = pd.read_csv(database_base_path + 'train.csv')
train = train[~train['image_id'].isin(imgs_to_remove)]
train.reset_index(inplace=True)
print('Train samples: %d' % len(train))

# Stratified K-Fold splitting for cross-validation
# Using StratifiedKFold to split the training data into N_FILES (5) folds.
# The split is stratified based on the labels to maintain class distribution in each fold.
folds = StratifiedKFold(n_splits=N_FILES, shuffle=True, random_state=seed)
train['file'] = -1

# Assign each sample to a fold
for fold_n, (train_idx, val_idx) in enumerate(folds.split(train, train['label'])):
    print('File: %s has %s samples' % (fold_n+1, len(val_idx)))
    train['file'].loc[val_idx] = fold_n

train.to_csv('train.csv', index=False)

# Creating TFRecord files by serializing image, label, and image name
for tfrec_num in range(N_FILES):
    print('\nWriting TFRecord %i of %i...' % (tfrec_num, N_FILES))
    samples = train[train['file'] == tfrec_num]
    n_samples = len(samples)
    print(f'{n_samples} samples')
    with tf.io.TFRecordWriter('Id_train%.2i-%i.tfrec' % (tfrec_num, n_samples)) as writer:
        for row in samples.itertuples():
            label = row.label
            image_name = row.image_id
            img_path = f'{PATH}{image_name}'
            img = cv2.imread(img_path)
            img = cv2.resize(img, (HEIGHT, WIDTH))
            img = cv2.imencode('.png', img, (cv2.IMWRITE_PNG_COMPRESSION, IMG_QUALITY))[1].tostring()
            example = serialize_example(img, label, str.encode(image_name))
            writer.write(example)

# Load and process the TFRecords, then display saved image samples for validation
# After creating TFRecords, this section loads them and counts the number of samples.
# It also saves samples for visual validation.
AUTO = tf.data.experimental.AUTOTUNE
FILENAMES = tf.io.gfile.glob('Id_train*.tfrec')
print(f'TFRecords files: {FILENAMES}')
print(f'Created image samples: {count_data_items(FILENAMES)}')

save_samples(load_dataset(FILENAMES, HEIGHT, WIDTH).batch(1), 6, 6)

# Define the classes for the images: HP (Hyperplastic Polyp) and SSA (Sessile Serrated Adenoma).
CLASSES = ['HP', 'SSA']

label_count = train.groupby('label', as_index=False).count()
label_count.rename(columns={'image_id': 'Count', 'label': 'Label'}, inplace=True)
label_count['Label'] = label_count['Label'].apply(lambda x: CLASSES[x])

# Plot the overall class distribution
fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax = sns.barplot(x=label_count['Count'], y=label_count['Label'], palette='viridis')
ax.tick_params(labelsize=16)

plt.savefig('All_label_count.png', bbox_inches='tight', dpi=300)



# Visualizing class distribution for each fold
# This loop generates a class distribution plot for each fold of the cross-validation.

for fold_n in range(folds.n_splits):
    label_count = train[train['file'] == fold_n].groupby('label', as_index=False).count()
    label_count.rename(columns={'image_id': 'Count', 'label': 'Label'}, inplace=True)
    
    # Add these two lines to print unique label indices and label_count DataFrame for each fold
    print(f'Unique label indices for fold {fold_n}:', label_count['Label'].unique())
    print(f'label_count DataFrame for fold {fold_n}:\n', label_count)
    
    # Modify this line to handle invalid indices
    label_count['Label'] = label_count['Label'].apply(lambda x: CLASSES[x] if 0 <= x < len(CLASSES) else f'Invalid index: {x}')
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    fig.suptitle(f'File {fold_n+1}', fontsize=22)
    ax = sns.barplot(x=label_count['Count'], y=label_count['Label'], palette='viridis')
    ax.tick_params(labelsize=16)
    # Save the plot as a PNG image with a dynamic filename
    plt.savefig(f'plot_file_{fold_n+1}.png', bbox_inches='tight', dpi=300)
    
