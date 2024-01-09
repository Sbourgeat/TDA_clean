#!/usr/bin/env python3

"""
This script applies density filtration on an input image using the Cubical Complex method.
The image is read from a specified directory and then processed to prepare it for filtration.
The filtration process involves optimizing the image using an inverse time decay learning rate
and stochastic gradient descent optimizer, with persistence loss and binary cross entropy loss.
The optimized image is then saved to a specified output directory.

Author: Samuel Bourgeat
"""


import numpy as np

from gtda.images import DensityFiltration
import tifffile as TIF
from tqdm import tqdm

from gudhi.tensorflow import CubicalLayer
import tensorflow as tf

INPUT_DIR = "/home/samuel/brainMorpho/Brains_to_CC/31_male.tif"
OUTPUT_DIR = "/home/samuel/brainMorpho/New_CC_Filtration/"

img = TIF.imread(INPUT_DIR)
img = img.max() - img
image = img / img.max()
X = image

print("Density filtration")
DF = DensityFiltration()
X_df = DF.fit_transform(X)
image = X_df / X_df.max()
Image_filtered = []

IM_NUMB = 1

for image_slice in tqdm(image):
    X_im = tf.Variable(
        initial_value=np.array(image_slice, dtype=np.float32), trainable=True
    )
    layer = CubicalLayer(homology_dimensions=[0])
    lr = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=1e-3, decay_steps=10, decay_rate=0.01
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    EP = 200

    for epoch in range(EP + 1):
        with tf.GradientTape() as tape:
            dgm = layer.call(X_im)[0][0]
            persistence_loss = 10 * tf.math.reduce_sum(tf.abs(dgm[:, 1] - dgm[:, 0]))
            BCE = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            bce = BCE(image_slice, X_im).numpy()
            loss = persistence_loss + bce

        gradients = tape.gradient(loss, [X_im])
        np.random.seed(epoch)
        gradients[0] = gradients[0] + np.random.normal(
            loc=0.0, scale=0.001, size=gradients[0].shape
        )
        optimizer.apply_gradients(zip(gradients, [X_im]))

    Image_filtered.append(X_im.numpy())
    IM_NUMB += 1

TIF.imwrite(
    OUTPUT_DIR + "31_male_cubedsegmented_wo_filtration_Nov2023.tif", Image_filtered
)
print("Done!")
