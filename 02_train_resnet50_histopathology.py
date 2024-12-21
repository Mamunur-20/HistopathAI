# Importing required libraries for mathematical operations, file handling, plotting, and model building
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
import efficientnet.tfkeras as efn
import tensorflow_addons as tfa
from sklearn.manifold import TSNE
from tensorflow.keras import mixed_precision
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import set_global_policy
# Setting seeds for reproducibility
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Initialize the random seed
seed = 0
seed_everything(seed)
warnings.filterwarnings('ignore')

# Setting up the TensorFlow distributed strategy (GPU or CPU)
# This section checks for available GPUs and sets memory growth to prevent out-of-memory errors.
# If no GPU is found, it falls back to the default strategy.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)  # Avoids OOM errors by allowing dynamic memory growth
        print(f'Running on GPU {gpus}')
        strategy = tf.distribute.MirroredStrategy()  # Enables distributed training across GPUs
    except RuntimeError as e:
        print(e)
else:
    strategy = tf.distribute.get_strategy()  # Uses the default CPU strategy

# Determine the number of replicas (or devices) available for distributed training
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

# Enable mixed precision for faster training with reduced memory usage
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Model parameters: These are the key parameters used in the training process, such as batch size, learning rate, image dimensions, and the number of classes.
BATCH_SIZE = 32 * REPLICAS  # Batch size is scaled based on the number of replicas (GPUs)
LEARNING_RATE = 3e-5 * REPLICAS  # Scaled learning rate
EPOCHS_SCL = 110  # Number of epochs for supervised contrastive learning
EPOCHS = 100  # Total number of epochs for classification
HEIGHT = 512  # Image height
WIDTH = 512  # Image width
HEIGHT_RS = 512  # Resized image height
WIDTH_RS = 512  # Resized image width
CHANNELS = 3  # Number of image channels (RGB)
N_CLASSES = 2  # Number of classes (binary classification: HP, SSA)
N_FOLDS = 5  # Number of folds for cross-validation
FOLDS_USED = 5  # Number of folds actually used for training

# Function to count the total number of data items in the dataset by parsing filenames
def count_data_items(filenames):
    n = [int(re.compile(r'-([0-9]*)\.').search(filename).group(1)) for filename in filenames]
    return np.sum(n)

# Load dataset and paths
# The training data is loaded from a CSV file and the image paths and TFRecord paths are specified.
database_base_path = '/g/data/nk53/mr3328/bracs/mhist/'
train = pd.read_csv(f'{database_base_path}train.csv') 
print(f'Train samples: {len(train)}')


IMAGES_PATH = "/g/data/nk53/mr3328/bracs/mhist/train_images/"
TF_RECORDS_PATH = "/g/data/nk53/mr3328/bracs/mhist/train_tfrecords/"
FILENAMES_COMP = [os.path.join(TF_RECORDS_PATH, file) for file in os.listdir(TF_RECORDS_PATH) if file.endswith('.tfrec')]
TRAINING_FILENAMES = [os.path.join(TF_RECORDS_PATH, file) for file in os.listdir(TF_RECORDS_PATH) if file.endswith('.tfrec')]

# Count the total number of training images
NUM_TRAINING_IMAGES = count_data_items(FILENAMES_COMP)
print(f'Number of training images: {NUM_TRAINING_IMAGES}')

# Class labels: The two classes involved in the classification task for the MHIST dataset
CLASSES = ['HP', 'SSA']  # HP (Hyperplastic Polyp) and SSA (Sessile Serrated Adenoma)

# Data augmentation function to apply various transformations to the images

def data_augment(image, label):
    p_rotation = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_spatial = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_rotate = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_1 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_2 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_pixel_3 = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_shear = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_crop = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    # Apply shear transformation with a random probability
    if p_shear > .2:
        if p_shear > .6:
            image = transform_shear(image, HEIGHT, shear=20.)
        else:
            image = transform_shear(image, HEIGHT, shear=-20.)
    
    # Apply rotation transformation with a random probability
    if p_rotation > .2:
        if p_rotation > .6:
            image = transform_rotation(image, HEIGHT, rotation=45.)
        else:
            image = transform_rotation(image, HEIGHT, rotation=-45.)
    
    # Apply random flipping of images
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    # Apply transpose transformation with random probability
    if p_spatial > .75:
        image = tf.image.transpose(image)
    
    # Apply random 90-degree rotations
    if p_rotate > .75:
        image = tf.image.rot90(image, k=3)  # Rotate 270 degrees
    elif p_rotate > .5:
        image = tf.image.rot90(image, k=2)  # Rotate 180 degrees
    elif p_rotate > .25:
        image = tf.image.rot90(image, k=1)  # Rotate 90 degrees
    
    # Apply pixel-level transformations (saturation, contrast, brightness)
    if p_pixel_1 >= .4:
        image = tf.image.random_saturation(image, lower=.7, upper=1.3)
    if p_pixel_2 >= .4:
        image = tf.image.random_contrast(image, lower=.8, upper=1.2)
    if p_pixel_3 >= .4:
        image = tf.image.random_brightness(image, max_delta=.1)
    
    # Apply random cropping
    if p_crop > .6:
        if p_crop > .9:
            image = tf.image.central_crop(image, central_fraction=.5)
        elif p_crop > .8:
            image = tf.image.central_crop(image, central_fraction=.6)
        elif p_crop > .7:
            image = tf.image.central_crop(image, central_fraction=.7)
        else:
            image = tf.image.central_crop(image, central_fraction=.8)
    elif p_crop > .3:
        crop_size = tf.random.uniform([], int(HEIGHT*.6), HEIGHT, dtype=tf.int32)
        image = tf.image.random_crop(image, size=[crop_size, crop_size, CHANNELS])
    
    # Resize image to the target dimensions
    image = tf.image.resize(image, size=[HEIGHT, WIDTH])
    
    # Apply cutout augmentation to mask out random sections of the image
    if p_cutout > .5:
        image = data_augment_cutout(image)
    
    return image, label  # Return the augmented image and label



# Function to randomly rotate an image
def transform_rotation(image, height, rotation):
    """
    Apply a random rotation to an input image. The rotation angle is chosen randomly.
    
    Parameters:
    image -- input image of shape [height, width, 3]
    height -- height of the image
    rotation -- maximum rotation angle in degrees

    Returns:
    Rotated image of the same shape as input.
    """
    DIM = height
    XDIM = DIM % 2  # Fix for image dimensions with odd size
    rotation = rotation * tf.random.uniform([1], dtype='float32')  # Randomly sample a rotation value

    # Convert rotation angle from degrees to radians
    rotation = math.pi * rotation / 180.0

    # Define the rotation matrix
    c1 = tf.math.cos(rotation)
    s1 = tf.math.sin(rotation)
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    rotation_matrix = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])

    # Create a grid of destination pixel indices
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # Rotate the destination pixel grid to map onto the origin pixels
    idx2 = K.dot(rotation_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # Gather pixel values from the original image at the rotated positions
    idx3 = tf.stack([DIM // 2 - idx2[0], DIM // 2 - 1 + idx2[1]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])

# Function to randomly shear an image
def transform_shear(image, height, shear):
    """
    Apply a random shear transformation to an input image.
    
    Parameters:
    image -- input image of shape [height, width, 3]
    height -- height of the image
    shear -- maximum shear angle in degrees

    Returns:
    Sheared image of the same shape as input.
    """
    DIM = height
    XDIM = DIM % 2 
    shear = shear * tf.random.uniform([1], dtype='float32')
    shear = math.pi * shear / 180.0

    # Define the shear matrix
    one = tf.constant([1], dtype='float32')
    zero = tf.constant([0], dtype='float32')
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)
    shear_matrix = tf.reshape(tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])

    # Create a grid of destination pixel indices
    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
    z = tf.ones([DIM * DIM], dtype='int32')
    idx = tf.stack([x, y, z])

    # Shear the destination pixel grid onto the origin pixels
    idx2 = K.dot(shear_matrix, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

    # Gather pixel values from the original image at the sheared positions
    idx3 = tf.stack([DIM // 2 - idx2[0], DIM // 2 - 1 + idx2[1]])
    d = tf.gather_nd(image, tf.transpose(idx3))

    return tf.reshape(d, [DIM, DIM, 3])

# CutOut function to randomly mask sections of the image
def data_augment_cutout(image, min_mask_size=(int(HEIGHT * .1), int(HEIGHT * .1)), max_mask_size=(int(HEIGHT * .125), int(HEIGHT * .125))):
    """
    Apply CutOut augmentation, which masks random parts of the image.
    
    Parameters:
    image -- input image of shape [height, width, 3]
    min_mask_size -- minimum size of the mask to apply
    max_mask_size -- maximum size of the mask to apply

    Returns:
    Image with random CutOut applied.
    """
    p_cutout = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    
    if p_cutout > .85:  # Apply between 10-15 cutouts
        n_cutout = tf.random.uniform([], 10, 15, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    elif p_cutout > .6:  # Apply between 5-10 cutouts
        n_cutout = tf.random.uniform([], 5, 10, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    elif p_cutout > .25:  # Apply between 2-5 cutouts
        n_cutout = tf.random.uniform([], 2, 5, dtype=tf.int32)
        image = random_cutout(image, HEIGHT, WIDTH, min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=n_cutout)
    else:  # Apply 1 cutout
        image = random_cutout(image, HEIGHT, WIDTH, min_mask_size=min_mask_size, max_mask_size=max_mask_size, k=1)

    return image

# Helper function to apply multiple random CutOuts
def random_cutout(image, height, width, channels=3, min_mask_size=(10, 10), max_mask_size=(80, 80), k=1):
    """
    Perform random CutOut augmentation with multiple masks.
    
    Parameters:
    image -- input image of shape [height, width, channels]
    height -- height of the image
    width -- width of the image
    channels -- number of image channels (usually 3 for RGB)
    min_mask_size -- minimum size of the cutout mask
    max_mask_size -- maximum size of the cutout mask
    k -- number of cutouts to apply

    Returns:
    Image with k random cutouts applied.
    """
    assert height > min_mask_size[0]
    assert width > min_mask_size[1]
    assert height > max_mask_size[0]
    assert width > max_mask_size[1]

    for i in range(k):
        mask_height = tf.random.uniform(shape=[], minval=min_mask_size[0], maxval=max_mask_size[0], dtype=tf.int32)
        mask_width = tf.random.uniform(shape=[], minval=min_mask_size[1], maxval=max_mask_size[1], dtype=tf.int32)

        pad_h = height - mask_height
        pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
        pad_bottom = pad_h - pad_top

        pad_w = width - mask_width
        pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
        pad_right = pad_w - pad_left

        cutout_area = tf.zeros(shape=[mask_height, mask_width, channels], dtype=tf.uint8)
        cutout_mask = tf.pad([cutout_area], [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=1)
        cutout_mask = tf.squeeze(cutout_mask, axis=0)
        image = tf.multiply(tf.cast(image, tf.float32), tf.cast(cutout_mask, tf.float32))

    return image

# Decode an image from PNG format
def decode_image(image_data):
    """
    Decode a PNG-encoded image to a uint8 tensor.
    
    Parameters:
    image_data -- PNG-encoded image data

    Returns:
    Decoded image tensor.
    """
    image = tf.image.decode_png(image_data, channels=3)
    return image

# Normalize image data (scale between 0 and 1)
def scale_image(image, label):
    """
    Cast tensor to float and normalize to range [0, 1].
    
    Parameters:
    image -- input image tensor
    label -- label corresponding to the image

    Returns:
    Normalized image and its label.
    """
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

# Resize and reshape images
def prepare_image(image, label):
    """
    Resize and reshape images to the expected size.
    
    Parameters:
    image -- input image tensor
    label -- label corresponding to the image

    Returns:
    Resized and reshaped image and its label.
    """
    image = tf.image.resize(image, [HEIGHT_RS, WIDTH_RS])
    image = tf.reshape(image, [HEIGHT_RS, WIDTH_RS, 3])
    return image, label

# Parse TFRecord examples
def read_tfrecord(example, labeled=True):
    """
    Parse data from a TFRecord example.
    
    Parameters:
    example -- serialized TFRecord example
    labeled -- flag to indicate if data is labeled

    Returns:
    Parsed image and label or image name.
    """
    if labeled:
        TFREC_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string), 
            'target': tf.io.FixedLenFeature([], tf.int64), 
        }
    else:
        TFREC_FORMAT = {
            'image': tf.io.FixedLenFeature([], tf.string), 
            'image_name': tf.io.FixedLenFeature([], tf.string), 
        }
    example = tf.io.parse_single_example(example, TFREC_FORMAT)
    image = decode_image(example['image'])
    if labeled:
        label_or_name = tf.cast(example['target'], tf.int32)
    else:
        label_or_name = example['image_name']
    return image, label_or_name

# Function to load and prepare dataset for training or inference
def get_dataset(FILENAMES, labeled=True, ordered=False, repeated=False, cached=False, augment=False):
    """
    Prepare the dataset for training or inference.
    
    Parameters:
    FILENAMES -- list of file paths for the dataset
    labeled -- whether the dataset contains labels
    ordered -- whether the data should be loaded in a specific order
    repeated -- whether the dataset should be repeated (for multiple epochs)
    cached -- whether the dataset should be cached for faster loading
    augment -- whether to apply data augmentation

    Returns:
    Prepared dataset.
    """
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False
        dataset = tf.data.Dataset.list_files(FILENAMES)
        dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=AUTO)
    else:
        dataset = tf.data.TFRecordDataset(FILENAMES, num_parallel_reads=AUTO)
        
    dataset = dataset.with_options(ignore_order)
    
    dataset = dataset.map(lambda x: read_tfrecord(x, labeled=labeled), num_parallel_calls=AUTO)
    
    if augment:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        
    dataset = dataset.map(scale_image, num_parallel_calls=AUTO)
    dataset = dataset.map(prepare_image, num_parallel_calls=AUTO)
    
    if not ordered:
        dataset = dataset.shuffle(2048)
    if repeated:
        dataset = dataset.repeat()
        
    dataset = dataset.batch(BATCH_SIZE)
    
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


# Visualization utility functions
# This section provides helper functions to visualize images, batches of images, and model performance.

np.set_printoptions(threshold=15, linewidth=80)

# Function to convert a batch of images and labels to numpy arrays
def batch_to_numpy_images_and_labels(data):
    """
    Convert a batch of images and labels to numpy arrays.
    
    Parameters:
    data -- a tuple containing a batch of images and labels

    Returns:
    Numpy arrays of images and labels. If no labels (test set), return None for labels.
    """
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object:  # If labels are binary strings (e.g., image IDs)
        numpy_labels = [None for _ in enumerate(numpy_images)]  # Replace labels with None
    return numpy_images, numpy_labels

# Function to create a title based on label and correct label (for predictions)
def title_from_label_and_target(label, correct_label):
    """
    Generate a title string based on predicted and correct labels.
    
    Parameters:
    label -- predicted label
    correct_label -- ground truth label

    Returns:
    Title string and a boolean indicating whether the prediction is correct.
    """
    if correct_label is None:
        return CLASSES[label], True  # For test data with no labels
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

# Function to display a single image with a title
def display_one_flower(image, title, subplot, red=False, titlesize=16):
    """
    Display one image in a subplot with a title.

    Parameters:
    image -- the image to be displayed
    title -- the title to display
    subplot -- the subplot configuration
    red -- whether to display the title in red (incorrect prediction)
    titlesize -- the font size for the title
    """
    plt.subplot(*subplot)
    plt.axis('off')  # Disable axis display
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), 
                  color='red' if red else 'black', 
                  fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)

# Function to display a batch of images
def display_batch_of_images(databatch, filename="batch_of_images.png", predictions=None):
    """
    Display a batch of images with optional predictions.

    Parameters:
    databatch -- batch of images (and optionally labels)
    filename -- name of the file to save the image
    predictions -- predicted labels for the batch (optional)
    """
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    rows = int(math.sqrt(len(images)))  # Create a square grid layout for displaying images
    cols = len(images) // rows
        
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)

    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols, FIGSIZE))
    
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows, cols)*40+3
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.savefig(filename, dpi=300)
    plt.close()

# Function to extract and convert a dataset to numpy arrays for visualization
def dataset_to_numpy_util(dataset, N):
    """
    Convert a TensorFlow dataset to numpy arrays for visualization.

    Parameters:
    dataset -- the dataset to be converted
    N -- number of images to extract

    Returns:
    Numpy arrays of images and labels.
    """
    dataset = dataset.unbatch().batch(N)
    for images, labels in dataset:
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()
        break
    return numpy_images, numpy_labels

# Function to create a title string for image evaluation based on predicted and correct labels
def title_from_label_and_target(label, correct_label):
    """
    Generate a title string for evaluation based on predicted and correct labels.
    
    Parameters:
    label -- predicted label (one-hot encoded)
    correct_label -- ground truth label

    Returns:
    Title string and a boolean indicating whether the prediction is correct.
    """
    # Convert the one-hot encoded label to its class index
    label = np.argmax(label, axis=-1)
    
    # Determine if the prediction is correct
    correct = (label == correct_label)
    
    # Create the title string with the predicted label and correctness status
    return "{} [{}{}{}]".format(
        label, 
        str(correct), 
        ', should be ' if not correct else '', 
        correct_label if not correct else ''
    ), correct

# Function to display a single image during evaluation
def display_one_flower_eval(image, title, subplot, red=False):
    """
    Display a single image with a title during evaluation.
    
    Parameters:
    image -- the image to be displayed
    title -- the title to display (showing predicted and actual labels)
    subplot -- the subplot configuration
    red -- whether to display the title in red (for incorrect predictions)
    
    Returns:
    Updated subplot index.
    """
    # Display the image in the specified subplot
    plt.subplot(subplot)
    plt.axis('off')  # Disable axis display
    plt.imshow(image)  # Show the image
    
    # Set the title, with red color if the prediction is incorrect
    plt.title(title, fontsize=14, color='red' if red else 'black')
    
    # Return the updated subplot index
    return subplot + 1



# Function to display 9 images with their predictions
def display_9_images_with_predictions(images, predictions, labels):
    """
    Display 9 images along with their predictions.

    Parameters:
    images -- the images to be displayed
    predictions -- predicted labels
    labels -- ground truth labels
    """
    subplot = 331
    plt.figure(figsize=(13, 13))
    for i, image in enumerate(images):
        title, correct = title_from_label_and_target(predictions[i], labels[i])
        subplot = display_one_flower_eval(image, title, subplot, not correct)
        if i >= 8:
            break

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig("predictions.png", dpi=300)
    plt.close()

# Function to plot training metrics like loss and accuracy
def plot_metrics(history, filename="metrics.png"):
    """
    Plot training metrics (loss and accuracy).

    Parameters:
    history -- training history containing loss and accuracy values
    filename -- name of the file to save the plot
    """
    fig, axes = plt.subplots(2, 1, sharex='col', figsize=(20, 8))
    axes = axes.flatten()

    axes[0].plot(history['loss'], label='Train loss')
    axes[0].plot(history['val_loss'], label='Validation loss')
    axes[0].legend(loc='best', fontsize=16)
    axes[0].set_title('Loss')
    axes[0].axvline(np.argmin(history['loss']), linestyle='dashed')
    axes[0].axvline(np.argmin(history['val_loss']), linestyle='dashed', color='orange')

    axes[1].plot(history['sparse_categorical_accuracy'], label='Train accuracy')
    axes[1].plot(history['val_sparse_categorical_accuracy'], label='Validation accuracy')
    axes[1].legend(loc='best', fontsize=16)
    axes[1].set_title('Accuracy')
    axes[1].axvline(np.argmax(history['sparse_categorical_accuracy']), linestyle='dashed')
    axes[1].axvline(np.argmax(history['val_sparse_categorical_accuracy']), linestyle='dashed', color='orange')

    plt.xlabel('Epochs', fontsize=16)
    sns.despine()
    plt.savefig(filename, dpi=300)
    plt.close()

# Function to visualize embeddings using TSNE
def visualize_embeddings(embeddings, labels, filename="embeddings.png", figsize=(16, 16)):
    """
    Visualize embeddings using TSNE.

    Parameters:
    embeddings -- the embeddings to be visualized
    labels -- the labels corresponding to the embeddings
    filename -- name of the file to save the plot
    figsize -- size of the plot
    """
    embed2D = TSNE(n_components=2, n_jobs=-1, random_state=seed).fit_transform(embeddings)
    embed2D_x = embed2D[:, 0]
    embed2D_y = embed2D[:, 1]

    df_embed = pd.DataFrame({'labels': labels})
    df_embed = df_embed.assign(x=embed2D_x, y=embed2D_y)

    df_embed_pb = df_embed[df_embed['labels'] == 0]
    df_embed_dcic = df_embed[df_embed['labels'] == 1]

    plt.figure(figsize=figsize)
    plt.scatter(df_embed_pb['x'], df_embed_pb['y'], color='yellow', s=10, label='HP')
    plt.scatter(df_embed_dcic['x'], df_embed_dcic['y'], color='blue', s=10, label='SSA')
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()

# Example usage: visualize batch of images
train_dataset = get_dataset(FILENAMES_COMP, ordered=True, augment=True)
train_iter = iter(train_dataset.unbatch().batch(20))

display_batch_of_images(next(train_iter))
display_batch_of_images(next(train_iter))

# Plot dataset distribution
ds_dist = get_dataset(FILENAMES_COMP)
labels_comp = [target.numpy() for img, target in iter(ds_dist.unbatch())]

fig, ax = plt.subplots(1, 1, figsize=(18, 8))
ax = sns.countplot(y=labels_comp, palette='viridis')
ax.tick_params(labelsize=16)
plt.savefig('dataset_distribution.png', dpi=300)
plt.close()

# Initialize lists to store performance metrics across folds
precisions = []
recalls = []
f1_scores = []
accuracies = []

# Initialize lists to store training/validation accuracy and loss for visualization
all_train_acc = []
all_val_acc = []
all_train_loss = []
all_val_loss = []

# Learning rate parameters for cosine learning rate scheduling
lr_start = 1e-8
lr_min = 1e-8
lr_max = LEARNING_RATE
num_cycles = 1.0  # Number of cosine cycles
warmup_epochs = 1  # Number of epochs for learning rate warmup
hold_max_epochs = 0  # Number of epochs to hold the maximum learning rate
total_epochs = EPOCHS
warmup_steps = warmup_epochs * (NUM_TRAINING_IMAGES // BATCH_SIZE)  # Calculate warmup steps
hold_max_steps = hold_max_epochs * (NUM_TRAINING_IMAGES // BATCH_SIZE)  # Calculate steps for holding max LR
total_steps = total_epochs * (NUM_TRAINING_IMAGES // BATCH_SIZE)  # Total steps across all epochs

# Cosine learning rate scheduler with warmup and optional holding of max learning rate
@tf.function
def cosine_schedule_with_warmup(step, total_steps, warmup_steps=0, hold_max_steps=0, 
                                lr_start=1e-4, lr_max=1e-3, lr_min=None, num_cycles=0.5):
    """
    Custom cosine learning rate schedule with warmup and holding max LR.

    Parameters:
    - step: Current training step
    - total_steps: Total number of training steps
    - warmup_steps: Number of warmup steps
    - hold_max_steps: Number of steps to hold max LR
    - lr_start: Starting learning rate
    - lr_max: Maximum learning rate
    - lr_min: Minimum learning rate
    - num_cycles: Number of cosine cycles

    Returns:
    - Adjusted learning rate based on the current step
    """
    if step < warmup_steps:
        lr = (lr_max - lr_start) / warmup_steps * step + lr_start
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = lr_max * (0.5 * (1.0 + tf.math.cos(np.pi * ((num_cycles * progress) % 1.0))))
        if lr_min is not None:
            lr = tf.math.maximum(lr_min, float(lr))
    return lr

# Generate the learning rate schedule over the total training steps
rng = [i for i in range(int(total_steps))]
y = [cosine_schedule_with_warmup(tf.cast(x, tf.float32), tf.cast(total_steps, tf.float32), 
                                 tf.cast(warmup_steps, tf.float32), hold_max_steps, 
                                 lr_start, lr_max, lr_min, num_cycles) for x in rng]

# Plot and save the learning rate schedule
sns.set(style='whitegrid')
fig, ax = plt.subplots(figsize=(20, 6))
plt.plot(rng, y)
plt.savefig("plot_learning.png", dpi=300)
plt.close()

# Print summary of total steps and learning rate schedule
print(f'{total_steps} total steps and {NUM_TRAINING_IMAGES // BATCH_SIZE} steps per epoch')
print(f'Learning rate schedule: {y[0]:.3g} to {max(y):.3g} to {y[-1]:.3g}')



# Set global precision policy to 'float32' for stability during training
set_global_policy('float32')

# Function to build an encoder model using EfficientNetB3
def encoder_fn(input_shape):
    """
    Builds the ResNet50 encoder.

    Args:
        input_shape: Tuple specifying the input image shape (height, width, channels).

    Returns:
        model: A ResNet50-based encoder model.
    """
    inputs = L.Input(shape=input_shape, name='inputs')
    base_model = ResNet50(weights="imagenet", include_top=False, pooling='avg')
    x = base_model(inputs)
    model = Model(inputs=inputs, outputs=x)
    return model


# Function to build a classifier on top of the encoder
def classifier_fn(input_shape, N_CLASSES, encoder, trainable=True):
    """
    Build and return a classification model on top of the encoder.

    Parameters:
    - input_shape: Shape of the input image
    - N_CLASSES: Number of output classes
    - encoder: Pre-built encoder model
    - trainable: Whether to allow training of the encoder layers

    Returns:
    - Model: A classification model
    """
    for layer in encoder.layers:
        layer.trainable = trainable 
    
    inputs = L.Input(shape=input_shape, name='inputs')
    features = encoder(inputs)
    features = L.Dropout(0.5)(features)  # Apply dropout for regularization
    features = L.Dense(1000, activation='relu')(features)  # Add fully connected layer
    features = L.Dropout(0.5)(features)  # Apply another dropout
    outputs = L.Dense(N_CLASSES, activation='softmax', name='outputs', dtype='float32')(features)  # Output layer

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Custom supervised contrastive loss function
temperature = 0.1
class SupervisedContrastiveLoss(losses.Loss):
    """
    Custom supervised contrastive loss function for training.

    Parameters:
    - temperature: Scaling factor for the contrastive loss
    """
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute similarity logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

# Function to add a projection head to the encoder for contrastive learning
def add_projection_head(input_shape, encoder):
    """
    Add a projection head to the encoder for contrastive learning.

    Parameters:
    - input_shape: Shape of the input image
    - encoder: Pre-built encoder model

    Returns:
    - Model: An encoder with an added projection head
    """
    inputs = L.Input(shape=input_shape, name='inputs')
    features = encoder(inputs)
    outputs = L.Dense(128, activation='relu', name='projection_head', dtype='float32')(features)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Ensure that the TFRecord path exists before proceeding
if not os.path.exists(TF_RECORDS_PATH):
    print(f"Error: {TF_RECORDS_PATH} does not exist.")



# Load and count the number of training images from the TFRecord files
FILENAMES_COMP = tf.io.gfile.glob(os.path.join(TF_RECORDS_PATH, '*.tfrec'))
NUM_TRAINING_IMAGES = count_data_items(FILENAMES_COMP)
print(f'Number of training images: {NUM_TRAINING_IMAGES}')


# K-Fold Cross-Validation Setup
skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
oof_reg_pred = []
oof_reg_labels = []
oof_reg_embed = []

# Loop through each fold for training and validation
for fold, (idxT, idxV) in enumerate(skf.split(np.arange(5))):
    if fold >= FOLDS_USED:
        break

    print(f'\nFOLD: {fold+1}')
    print(f'TRAIN: {idxT} VALID: {idxV}')

    # Fetch the training and validation TFRecord filenames for the current fold
    TRAIN_FILENAMES = tf.io.gfile.glob([os.path.join(TF_RECORDS_PATH, f'Id_train{i:02d}*.tfrec') for i in idxT])
    VALID_FILENAMES = tf.io.gfile.glob([os.path.join(TF_RECORDS_PATH, f'Id_train{i:02d}*.tfrec') for i in idxV])
    print(f'TRAIN FILENAMES: {TRAIN_FILENAMES}')
    print(f'VALID FILENAMES: {VALID_FILENAMES}')

    np.random.shuffle(TRAIN_FILENAMES)
    ct_train = count_data_items(TRAIN_FILENAMES)
    step_size = ct_train // BATCH_SIZE
    total_steps = EPOCHS * step_size

    # Build and compile the ResNet50 model
    with strategy.scope():
        encoder_reg = encoder_fn((None, None, CHANNELS))
        model_reg = classifier_fn((None, None, CHANNELS), N_CLASSES, encoder_reg)
        model_reg.summary()

        optimizer = optimizers.Adam()
        lr = lambda: cosine_schedule_with_warmup(
            tf.cast(optimizer.iterations, tf.float32),
            tf.cast(total_steps, tf.float32),
            tf.cast(warmup_steps, tf.float32),
            hold_max_steps, lr_start, lr_max, lr_min, num_cycles
        )
        optimizer = optimizers.Adam(learning_rate=lr)
        model_reg.compile(optimizer=optimizer,
                          loss=losses.SparseCategoricalCrossentropy(),
                          metrics=[metrics.SparseCategoricalAccuracy()])

    # Train the model
    tf.config.run_functions_eagerly(True)
    history_reg = model_reg.fit(
        x=get_dataset(TRAIN_FILENAMES, repeated=True, augment=True),
        validation_data=get_dataset(VALID_FILENAMES, ordered=True, cached=True),
        steps_per_epoch=step_size,
        epochs=EPOCHS,
        verbose=2
    ).history

    # Save model weights for the current fold
    model_path = f'model_reg_{fold}.h5'
    model_reg.save_weights(model_path)

    print(f"#### FOLD {fold+1} OOF Accuracy = {np.max(history_reg['val_sparse_categorical_accuracy']):.5f}")

    # Out-of-Fold (OOF) Predictions and Embeddings
    ds_valid = get_dataset(VALID_FILENAMES, ordered=True)
    oof_reg_labels.append([target.numpy() for img, target in iter(ds_valid.unbatch())])
    x_oof = ds_valid.map(lambda image, target: image)
    oof_reg_pred.append(np.argmax(model_reg.predict(x_oof), axis=-1))
    oof_reg_embed.append(encoder_reg.predict(x_oof))

    # Classification Report for the Current Fold
    y_true_fold = oof_reg_labels[-1]
    y_pred_fold = oof_reg_pred[-1]

    print(f"Classification Report for Fold {fold+1}:")
    print(classification_report(y_true_fold, y_pred_fold, target_names=CLASSES))

    # Collect fold-level metrics
    report = classification_report(y_true_fold, y_pred_fold, target_names=CLASSES, output_dict=True)
    precisions.append(report['macro avg']['precision'])
    recalls.append(report['macro avg']['recall'])
    f1_scores.append(report['macro avg']['f1-score'])
    accuracies.append(report['accuracy'])

    all_train_acc.append(history_reg['sparse_categorical_accuracy'])
    all_val_acc.append(history_reg['val_sparse_categorical_accuracy'])
    all_train_loss.append(history_reg['loss'])
    all_val_loss.append(history_reg['val_loss'])

# Overall Metrics Across All Folds
mean_precision = np.mean(precisions)
std_precision = np.std(precisions)

mean_recall = np.mean(recalls)
std_recall = np.std(recalls)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print(f"\nOverall Metrics Across All Folds:")
print(f"Precision: {mean_precision:.5f} ± {std_precision:.5f}")
print(f"Recall: {mean_recall:.5f} ± {std_recall:.5f}")
print(f"F1 Score: {mean_f1:.5f} ± {std_f1:.5f}")
print(f"Accuracy: {mean_accuracy:.5f} ± {std_accuracy:.5f}")

# Plot Training and Validation Metrics Across All Folds
plt.figure(figsize=(12, 6))
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

for fold in range(FOLDS_USED):
    color = colors[fold % len(colors)]
    plt.plot(all_train_acc[fold], color, label=f'Train Acc Fold {fold+1}')
    plt.plot(all_val_acc[fold], color + '--', label=f'Valid Acc Fold {fold+1}')
    plt.plot(all_train_loss[fold], color + ':', label=f'Train Loss Fold {fold+1}')
    plt.plot(all_val_loss[fold], color + '-.', label=f'Valid Loss Fold {fold+1}')

plt.title('Training and Validation Metrics Across All Folds')
plt.xlabel('Epochs')
plt.ylabel('Metric Value')
plt.legend(loc='upper left', bbox_to_anchor=(1,1))
plt.tight_layout()

# Save the figure
plt.savefig('training_validation_metrics.png', dpi=300)
plt.close()

# Embedding Visualization and Confusion Matrix
y_reg_true = np.concatenate(oof_reg_labels)
y_reg_pred = np.concatenate(oof_reg_pred)
embeddings_reg = np.concatenate(oof_reg_embed)

# Visualize embeddings using TSNE
visualize_embeddings(embeddings_reg, y_reg_true, filename="my_plot_regular_resnet.png")


# Confusion Matrix
fig, ax = plt.subplots(1, 1, figsize=(20, 12))
cfn_matrix = confusion_matrix(y_reg_true, y_reg_pred, labels=range(len(CLASSES)))
cfn_matrix = (cfn_matrix.T / cfn_matrix.sum(axis=1)).T
df_cm = pd.DataFrame(cfn_matrix, index=CLASSES, columns=CLASSES)
ax = sns.heatmap(df_cm, cmap='Blues', annot=True, fmt='.2f', linewidths=.5).set_title('OOF', fontsize=30)

# Save the plot to a file
plt.savefig("confusion_matrix_resnet_regular.png", dpi=300)
plt.close()

# Classification report for all folds
print(classification_report(y_reg_true, y_reg_pred, target_names=CLASSES))
