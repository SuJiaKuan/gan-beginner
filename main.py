import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

from networks import generator
from networks import discriminator


Z_DIMENSIONS = 100
BATCH_SIZE = 50


def main():
    # Load mnist data.
    mnist = input_data.read_data_sets('MNIST_data/')

    # The placehodler for feeding input noise to the generator.
    z_placeholder = tf.placeholder(tf.float32, [None, Z_DIMENSIONS], name='z_placeholder')
    # The placehodler for feeding input images to the discriminator.
    x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x_placeholder')

    # The generated images.
    Gz = generator(z_placeholder, BATCH_SIZE, Z_DIMENSIONS)
    # The discriminator prediction probability for the real images.
    Dx = discriminator(x_placeholder)
    # The discriminator prediction probability for the generated images.
    Dg = discriminator(Gz, reuse_variables=True)

    # Two Loss Functions for discriminator.
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dx, labels = tf.ones_like(Dx)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.zeros_like(Dg)))
    # Loss function for generator.
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = Dg, labels = tf.ones_like(Dg)))

    # Get the varaibles for different network.
    tvars = tf.trainable_variables()
    d_vars = [var for var in tvars if 'd_' in var.name]
    g_vars = [var for var in tvars if 'g_' in var.name]

    # Train the discriminator.
    d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
    d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)
    # Train the generator.
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

    # From this point forward, reuse variables.
    tf.get_variable_scope().reuse_variables()

    with tf.Session() as sess:
        # Send summary statistics to TensorBoard.
        tf.summary.scalar('Generator_loss', g_loss)
        tf.summary.scalar('Discriminator_loss_real', d_loss_real)
        tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)

        images_for_tensorboard = generator(z_placeholder, BATCH_SIZE, Z_DIMENSIONS)
        tf.summary.image('Generated_images', images_for_tensorboard, 5)
        merged = tf.summary.merge_all()
        logdir = 'tensorboard/{}/'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        writer = tf.summary.FileWriter(logdir, sess.graph)

        sess.run(tf.global_variables_initializer())

        # Pre-train discriminator.
        for i in range(300):
            z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMENSIONS])
            real_image_batch = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE, 28, 28, 1])
            _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                   {x_placeholder: real_image_batch, z_placeholder: z_batch})

            if i % 100 == 0:
                print("dLossReal:", dLossReal, "dLossFake:", dLossFake)

        # Train generator and discriminator together
        for i in range(100000):
            real_image_batch = mnist.train.next_batch(BATCH_SIZE)[0].reshape([BATCH_SIZE, 28, 28, 1])
            z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMENSIONS])

            # Train discriminator on both real and fake images.
            _, __, dLossReal, dLossFake = sess.run([d_trainer_real, d_trainer_fake, d_loss_real, d_loss_fake],
                                                   {x_placeholder: real_image_batch, z_placeholder: z_batch})

            # Train generator.
            z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMENSIONS])
            _ = sess.run(g_trainer, feed_dict={z_placeholder: z_batch})

            if i % 10 == 0:
                # Update TensorBoard with summary statistics.
                z_batch = np.random.normal(0, 1, size=[BATCH_SIZE, Z_DIMENSIONS])
                summary = sess.run(merged, {z_placeholder: z_batch, x_placeholder: real_image_batch})
                writer.add_summary(summary, i)


if __name__ == '__main__':
    main()
