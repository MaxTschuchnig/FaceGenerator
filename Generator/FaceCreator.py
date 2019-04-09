# Sources: https://github.com/naokishibuya/deep-learning/blob/master/python/dcgan_celeba.ipynb

from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import matplotlib.pyplot as plt


def save_images(generated_images, filename):
    n_images = len(generated_images)
    for i in range(n_images):
        img = deprocess(generated_images[i])
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(filename + str(i))
        plt.close('all')


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
    generator.load_weights('gen_weights_1.h5')

    # discriminator
    discriminator = make_discriminator(leaky_alpha, init_std)
    discriminator.compile(optimizer=Adam(lr=d_learning_rate, beta_1=d_beta_1), loss='binary_crossentropy')

    # set discriminator now untrainable, already compiled disc will stay trainable but
    # discriminator in GAN will be untrainable, see: https://github.com/keras-team/keras/issues/4674
    make_trainable(discriminator, False)
    discriminator.load_weights('dis_weights_1.h5')

    # GAN
    gan = Sequential([generator, discriminator])
    gan.compile(optimizer=Adam(lr=g_learning_rate, beta_1=g_beta_1), loss='binary_crossentropy')
    gan.load_weights('gan_weights_1.h5')

    return gan, generator, discriminator


# Prepare training
def make_latent_samples (n_samples, sample_size):
    return np.random.normal(loc=0, scale=1, size=(n_samples, sample_size))


def make_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable


def make_labels(size):
    return np.ones([size, 1]), np.zeros([size, 1])


# As we will use a tanh activation for the generator, the image data should be in a range of -1 to 1.
# These functions accomplish this conversion.
def preprocess(x):
    # as we use 8bit color images
    return (x/255)*2-1


def deprocess(x):
    # as we use 8bit color images. If uint8 is not used, the image will be in a wrong color format.
    return np.uint8((x+1)/2*255)


g_learning_rate = 0.0001
g_beta_1 = 0.5
d_learning_rate = 0.001
d_beta_1 = 0.5
leaky_alpha = 0.2
init_std = 0.02
sample_size = 100

gan, generator, discriminator = make_DCGAN(
        sample_size,
        g_learning_rate,
        g_beta_1,
        d_learning_rate,
        d_beta_1,
        leaky_alpha,
        init_std
    )

save_images(generator.predict(make_latent_samples(1000, sample_size)), 'generated_')
