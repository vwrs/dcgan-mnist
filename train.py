#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
from keras.datasets import mnist
from PIL import Image
from model import discriminator, generator
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Activation
from keras.layers.advanced_activations import ELU
from visualizer import *

BATCH_SIZE = 32
NUM_EPOCH = 100
LR = 0.0002  # initial learning rate
B1 = 0.5  # momentum term
GENERATED_IMAGE_PATH = 'images/'
GENERATED_MODEL_PATH = 'models/'

def train():
    (X_train, y_train), (_, _) = mnist.load_data()
    # normalize images
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)

    g = generator()
    d = discriminator(activation=ELU())

    opt = Adam(lr=LR,beta_1=B1)
    d.trainable = True
    d.compile(loss='binary_crossentropy',
              metrics=['accuracy'],
              optimizer=opt)
    d.trainable = False
    dcgan = Sequential([g, d])
    opt= Adam(lr=LR,beta_1=B1)
    dcgan.compile(loss='binary_crossentropy',
                  metrics=['accuracy'],
                  optimizer=opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    # create directory
    if not os.path.exists(GENERATED_IMAGE_PATH):
        os.mkdir(GENERATED_IMAGE_PATH)
    # if not os.path.exists(GENERATED_MODEL_PATH):
    #     os.mkdir(GENERATED_MODEL_PATH)

    print("-------------------")
    print('Number of batches:', num_batches)
    print("-------------------")
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            X_g = np.array([np.random.normal(0,0.5,100) for _ in range(BATCH_SIZE)])
            y_g = [1]*BATCH_SIZE
            noise = np.array([np.random.normal(0,0.5,100) for _ in range(BATCH_SIZE)])
            generated_images = g.predict(noise, verbose=0)

            X_d_true = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            X_d_gen  = generated_images
            y_d_true = [1]*BATCH_SIZE
            y_d_gen = [0]*BATCH_SIZE

            # train discriminator
            d_loss = d.train_on_batch(X_d_true, y_d_true)
            d_loss = d.train_on_batch(X_d_gen, y_d_gen)
            # train generator
            g_loss = dcgan.train_on_batch(X_g, y_g)
            sys.stdout.write("\repoch: %d, batch: %d, g_loss: %f, d_loss: %f, g_accuracy: %f, d_accuracy: %f" %
                    (epoch, index, g_loss[0], d_loss[0], g_loss[1], d_loss[1]))
            sys.stdout.flush()

        # output generated image
        image = combine_images(generated_images)
        image = image*127.5 + 127.5
        Image.fromarray(image.astype(np.uint8))\
            .save(GENERATED_IMAGE_PATH+"%03depoch.png" % (epoch))
        print()
        # g.save(GENERATED_MODEL_PATH+'dcgan_generator.h5')
        # d.save(GENERATED_MODEL_PATH+'dcgan_discriminator.h5')

if __name__ == '__main__':
    train()
