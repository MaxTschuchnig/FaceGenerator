# Sources: https://github.com/naokishibuya/deep-learning/blob/master/python/dcgan_celeba.ipynb

from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import sys


def plot_first_20_images(_filenames):
    plt.figure(figsize=(10,8))
    for i in range(20):
        img = plt.imread(_filenames[i])
        plt.subplot(4, 5, i+1)
        plt.imshow(img)
        plt.title(img.shape)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


# Defining preprocessing functions (mainly in order to ease the training steps, time)
def load_image(_filename, size=(64, 64)):
    # def load_image(_filename, size=(32, 32)):
    img = plt.imread(_filename)

    # cropping of image
    rows, cols = img.shape[:2]
    # for calculating the first and last pixel of the current crop
    crop_r, crop_c = 150, 150
    start_row, start_col = (rows - crop_r) // 2, (cols - crop_c) // 2
    end_row, end_col = rows - start_row, cols - start_row
    img = img[start_row:end_row, start_col:end_col, :]

    # resize
    img = imresize(img, size)

    return img

def plot_first_20_reduced_images(_filenames):
    plt.figure(figsize=(10,8))
    for i in range(20):
        img = load_image(_filenames[i])
        plt.subplot(4, 5, i+1)
        plt.imshow(img)
        plt.title(img.shape)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


# Generator
def make_generator(input_size, leaky_alpha, init_stddev):
    # generates images in (32,32,3)
    return Sequential([
        Dense(8*8*256, input_shape=(input_size,),
        # Dense(4 * 4 * 512, input_shape=(input_size,),
              kernel_initializer=RandomNormal(stddev=init_stddev)),
        Reshape(target_shape=(8, 8, 256)),
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),

        Conv2DTranspose(128, kernel_size=5, strides=2, padding='same',
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 8x8
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),

        Conv2DTranspose(64, kernel_size=5, strides=2, padding='same',
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 16x16
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),

        Conv2DTranspose(3, kernel_size=5, strides=2, padding='same',
                        kernel_initializer=RandomNormal(stddev=init_stddev)), # 32x32
        Activation('tanh')
    ])


# Discriminator
def make_discriminator(leaky_alpha, init_stddev):
    # classifies images in (64,64,3)
    return Sequential([
        Conv2D(64, kernel_size=5, strides=2, padding='same',
               kernel_initializer=RandomNormal(stddev=init_stddev),    # 16x16
               input_shape=(64, 64, 3)),
               # input_shape=(32, 32, 3)),
        LeakyReLU(alpha=leaky_alpha),

        Conv2D(128, kernel_size=5, strides=2, padding='same',
               kernel_initializer=RandomNormal(stddev=init_stddev)),   # 8x8
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),

        Conv2D(256, kernel_size=5, strides=2, padding='same',
               kernel_initializer=RandomNormal(stddev=init_stddev)),   # 4x4
        BatchNormalization(),
        LeakyReLU(alpha=leaky_alpha),

        Flatten(),
        Dense(1, kernel_initializer=RandomNormal(stddev=init_stddev)),
        Activation('sigmoid')
    ])


# Combine into DCGAN
# beta_1 is the exponential decay rate for the 1st moment estimates in Adam optimizer
def make_DCGAN(sample_size,
               g_learning_rate,
               g_beta_1,
               d_learning_rate,
               d_beta_1,
               leaky_alpha,
               init_std):
    # generator
    generator = make_generator(sample_size, leaky_alpha, init_std)

    # discriminator
    discriminator = make_discriminator(leaky_alpha, init_std)
    discriminator.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta_1), loss='binary_crossentropy')

    # set discriminator now untrainable, already compiled disc will stay trainable but
    # discriminator in GAN will be untrainable, see: https://github.com/keras-team/keras/issues/4674
    make_trainable(discriminator, False)

    # GAN
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta_1), loss='binary_crossentropy')

    return gan, generator, discriminator


# Prepare training
def make_latent_samples (n_samples, sample_size):
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))


def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable


def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])


def show_losses(losses, filename, save2file=False):
    losses = np.array(losses)

    fig, ax = plt.subplots()
    plt.plot(losses.T[0], label='Discriminator')
    plt.plot(losses.T[1], label='Generator')
    plt.title("Validation Losses")
    plt.legend()

    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


def show_images(generated_images, filename, save2file=False):
    n_images = len(generated_images)
    cols = 10
    rows = n_images // cols

    plt.figure(figsize=(10, 8))
    for i in range(n_images):
        img = deprocess(generated_images[i])
        ax = plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()

    if save2file:
        plt.savefig(filename)
        plt.close('all')
    else:
        plt.show()


def train(
        c_name,
        g_learning_rate,    # generator learning rate
        g_beta_1,           # the exponential decay rate for the 1st moment estimation in the generator Adam optimizer
        d_learning_rate,    # discriminator learning rate
        d_beta_1,           # the exponential decay rate for the 1st moment estimation in the discriminator Adam optimizer
        leaky_alpha,
        init_std,
        smooth = 0.1,
        sample_size = 100,  # latent sample size (i.e. 100 random numbers)
        epochs = 250,
        batch_size = 128,   #
        eval_size = 32,     # evaluate size
        show_details = True):

    # generate labels from batch size
    y_train_real, y_train_fake = make_labels(batch_size)
    y_eval_real, y_eval_fake = make_labels(eval_size)

    # create a GAN, generator, discriminator
    gan, generator, discriminator = make_DCGAN(
        sample_size,
        g_learning_rate,
        g_beta_1,
        d_learning_rate,
        d_beta_1,
        leaky_alpha,
        init_std
    )

    losses = []
    end = False
    for e in range(epochs):
        for i in tqdm(range(len(X_train)//batch_size)):
            # real CelebA images
            # take the current batch
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            # and load every image from the batch, preprocessed
            X_batch_real = np.array([preprocess(load_image(filename)) for filename in X_batch])

            # Generate latent samples and generate images with generator
            latent_samples = make_latent_samples(batch_size, sample_size)
            X_batch_fake = generator.predict_on_batch(latent_samples)

            d_train = 0
            g_train = 0

            # Add a little smoothness to increase learning
            d_train += discriminator.train_on_batch(X_batch_real, y_train_real * (1 - smooth))
            d_train += discriminator.train_on_batch(X_batch_fake, y_train_fake)

            # since generator is first in gan we need to add the latent sample
            g_train += gan.train_on_batch(latent_samples, y_train_real)

            if (g_train < 0.0001 and d_train > 12) or (g_train > 12 and d_train < 0.0001):
                end = True
                break

        # eval
        X_eval = X_test[np.random.choice(len(X_test), eval_size, replace=False)]
        X_eval_real = np.array([preprocess(load_image(filename)) for filename in X_eval])

        latent_samples = make_latent_samples(eval_size, sample_size)
        X_eval_fake = generator.predict_on_batch(latent_samples)

        # calculate disc loss by checking real against real labels and fake against fake labels
        d_loss = discriminator.test_on_batch(X_eval_real, y_eval_real)
        d_loss += discriminator.test_on_batch(X_eval_fake, y_eval_fake)

        # Generate the loss against real data
        g_loss = gan.test_on_batch(latent_samples, y_eval_real)

        if end:
            print('####################################################')
            print('In local very strong local minima')
            print('D:' + str(d_loss) + ',G:' + str(g_loss))
            print('####################################################')
            sys.exit()

        # Append losses
        losses.append((d_loss, g_loss))

        print("Epoch: {:>3}/{} Discriminator Loss: {:>6.4f} Generator Loss: {:>6.4f}".format(
            e + 1, epochs, d_loss, g_loss))
        show_images(X_eval_fake[:10], 'epoch_' + str(e), True)

    # show the result
    if show_details:
        show_losses(losses, 'losses' + c_name, True)
        show_images(generator.predict(make_latent_samples(80, sample_size)), 'result' + c_name, True)

    generator.save_weights('gen_weights_' + c_name + '.h5')
    discriminator.save_weights('dis_weights_' + c_name + '.h5')
    gan.save_weights('gan_weights_' + c_name + '.h5')

    return generator



# As we will use a tanh activation for the generator, the image data should be in a range of -1 to 1.
# These functions accomplish this conversion.
def preprocess(x):
    # as we use 8bit color images
    return (x/255)*2-1


def deprocess(x):
    # as we use 8bit color images. If uint8 is not used, the image will be in a wrong color format.
    return np.uint8((x+1)/2*255)

#
# Download the dataset by following the link and unzipping
#
#!wget https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip
#!unzip celeba.zip

# The dataset is big so we load all the filenames and load images in batch when needed.
filenames = np.array(glob('../FaceCreationData/*.jpg'))
# print(filenames)

X_train, X_test = train_test_split(filenames, test_size=1000)

# Plot the first 20 images for EDA
# plot_first_20_images(filenames)

# Preprocessing
# plot_first_20_reduced_images(filenames)
# Therefore, now we can load the dataset in a calculable fashion

train(sys.argv[1],
      g_learning_rate=float(sys.argv[2]),
      g_beta_1=float(sys.argv[3]),
      d_learning_rate=float(sys.argv[4]),
      d_beta_1=float(sys.argv[5]),
      leaky_alpha=float(sys.argv[6]),
      init_std=float(sys.argv[7]))
