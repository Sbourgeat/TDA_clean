########################################################
#													   #
#													   #
# This script is a code modified from ... 			   #
# by Samuel Bourgeat, Ph.D student, EPFL, Jaksic lab.  #
#													   #
#													   #
########################################################


#### import libraries

print("Importing packages")

import pyvista as pv
import gudhi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import velour

from sklearn.metrics import pairwise_distances

import plotly.express as px
from plotly.offline import init_notebook_mode, iplot


# gtda plotting functions
from gtda.plotting import plot_heatmap

from gtda.pipeline import Pipeline

from gtda.homology import WeakAlphaPersistence, VietorisRipsPersistence
from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, BettiCurve, PairwiseDistance
from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance
from gtda.homology import CubicalPersistence



from gtda.images import DensityFiltration


# image reading functions
print("Importing images")

from PIL import Image
import tifffile as TIF

import glob

# Tensorflow files

from tqdm                    import tqdm
from gudhi.tensorflow        import LowerStarSimplexTreeLayer, CubicalLayer, RipsLayer
import tensorflow            as tf


# import files

input_dir = "Cubical_complex/only_brain/"
output_dir ="Cubical_complex/Res/"

# Import the tiff files with tifffile

 
# assign directory
directory = '/home/samuel/brainMorpho/THE_SCANS/only_brain/Only_brains/'
 
# iterate over files in
# that directory
Files = []
for filename in glob.iglob(f'{directory}/*'):
    Files.append(filename)



##########################################################
#
#           Big iteration over all files
#
##########################################################

# Define Entropy variable 

Entropy_all = []

for file in Files:

    img = TIF.imread(file)

    #img = Image.open(options.input)
    img= img.max() - img
    image = img / img.max()
    #plt.imshow(image[120])



    # Reshape the image to fit the format needed (only if there is one image to analyse)
    print("Initializing images")

    X = image#[160]
    #X = X.reshape(1, *X.shape)

    # Increase the contrast between voxels in the images 
    print("Density filtration")

    DF = DensityFiltration()

    X_df = DF.fit_transform(X)


    # Compute cubical complex


    cubical_persistence = CubicalPersistence(n_jobs=-1)
    im_cubical = cubical_persistence.fit_transform(X_df)


    # Compute persistence entropy

    persistence_entropy = PersistenceEntropy()

    # calculate topological feature matrix
    X_basic = persistence_entropy.fit_transform(im_cubical)

    pd.DataFrame(X_basic).to_csv(output_dir+"PE_all_120160.csv", index=False)


    # Normalize the image to have a max value of 1 and min value 0 for each pixels

    image = X_df/X_df.max() 
    image = image[150:160]

    # Initialize the tensor cubical layer

    print("Initializing tensorflow")

    X = tf.Variable(initial_value=np.array(image, dtype=np.float32), trainable=True)
    layer = CubicalLayer(homology_dimensions=[0]) 


    # Initialize learning rate

    lr = tf.keras.optimizers.schedules.InverseTimeDecay(initial_learning_rate=1e-3, decay_steps=10, decay_rate=.01)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)


    # Run the optimisation

    ep = 2000


    losses, dgms = [], []
    for epoch in tqdm(range(ep+1)):
        with tf.GradientTape() as tape:
            dgm = layer.call(X)[0][0]
            # Squared distances to the diagonal
            persistence_loss = 15*tf.math.reduce_sum(tf.abs(dgm[:,1]-dgm[:,0])) # This value defines the max distance we allow to have between the diagonal and the points in the persistence diagrams 
            # 0-1 regularization for the pixels
            regularization = 0#tf.math.reduce_sum(tf.math.minimum(tf.abs(X),tf.abs(1-X)))
            loss = persistence_loss + regularization
        gradients = tape.gradient(loss, [X])
        
        # We also apply a small random noise to the gradient to ensure convergence
        np.random.seed(epoch)
        gradients[0] = gradients[0] + np.random.normal(loc=0., scale=.001, size=gradients[0].shape)
        
        optimizer.apply_gradients(zip(gradients, [X]))
        losses.append(loss.numpy())
        dgms.append(dgm)
        
        #plt.figure()
        #plt.imshow(X.numpy(), cmap='Greys')
        #plt.title('Image at epoch ' + str(epoch))
        #plt.show()



    # Save outputs 
    #losses.to_csv(output_dir + "Losses.csv")
    #dgsm.to_csv(output_dir + "dgms.csv")
    TIF.imwrite(output_dir + 'Image_output_all.tiff', X.numpy())

    # ploting results

    plt.figure()
    plt.imshow(X.numpy(), cmap='Greys')
    plt.title('Image at epoch ' + str(epoch))
    plt.show()



    plt.figure()
    plt.plot(losses)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()


    plt.figure()
    plt.scatter(dgms[0][:,0], dgms[0][:,1], s=40, marker='D', c='blue')
    for dg in dgms[:-1]:
        plt.scatter(dg[:,0], dg[:,1], s=20, marker='D', alpha=0.1)
    plt.scatter(dgms[-1][:,0], dgms[-1][:,1], s=40, marker='D', c='red')
    plt.plot([0,1], [0,1])
    plt.title('Optimized persistence diagrams')
    plt.show()












