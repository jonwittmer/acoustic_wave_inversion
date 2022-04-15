'''
   file:        classification.py
   author:      Jonathan Wittmer
   description: A starter script for working with deep neural networks 
                for a regression problem. The main tasks are
                  1. Load the dataset
                  2. Preprocess the dataset for input to DNN
                  3. Design DNN
                  4. Train DNN
                Provided are all the functions/function stubs required to 
                perform regression on MNIST dataset. The regression problem
                for this exercise will be to learn the identity map through a 
                DNN using an autoencoder. An autoencoder consists of 2 pieces:
                and encoder and a decoder. The encoder in this case reduces the 
                dimension of the input. The output of the encoder is often referred to 
                as the "latent space". The decoder portion of the network maps from the 
                latent space back to the original space. The goal is to reconstruct the 
                original image after compressing to the latent space. I like to think 
                of autoencoders like a converging-diverging nozzle. 

                     latent_space = encoder(input)
                     output       = decoder(latent_space)
                     error        = ||output - input||
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def normalizeData(data):
    min_val = tf.math.reduce_min(data)
    max_val = tf.math.reduce_max(data)
    data_range = tf.maximum(max_val - min_val, 1e-7)
    return (data - min_val) / data_range

def loadDataset(filenames, batch_size):
    # load data from files and concatenate into single numpy array
    data = np.load(filenames.pop())
    while filenames:
        data = np.concatenate((data, np.load(filenames.pop())))
    
    # Convert to tensorflow dataset.
    # Normalize data from physical units -> [0.0, 1.0]. Cache for performance.
    # Create batched dataset so we can iterate over the batches.
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(normalizeData).cache().shuffle(10000)
    dataset = tf.data.Dataset.zip((dataset, dataset))
    dataset = dataset.batch(batch_size)

    info = {
        'element_shape' : data.shape[1],
        'n_elements' : data.shape[0],
    }
    
    return dataset, info

def defineEncoder(latent_space_dimension, activation, dtype=tf.float64):
    # Create a list of keras layers.
    layers = [
        tf.keras.layers.Dense(1000, activation=activation, dtype=dtype),
        tf.keras.layers.Dense(256, activation=activation, dtype=dtype),
        tf.keras.layers.Dense(256, activation=activation, dtype=dtype),
        tf.keras.layers.Dense(256, activation=activation, dtype=dtype),
    ]
    
    layers.append(tf.keras.layers.Dense(latent_space_dimension, activation=activation, dtype=dtype))

    # To create a standard feedforward DNN, use Keras Sequential model
    model = tf.keras.models.Sequential(layers)
    return model

def defineDecoder(output_shape, activation, dtype=tf.float64):
    # Create a list of keras layers.
    layers = [
        tf.keras.layers.Dense(128, activation=activation, dtype=dtype),
        tf.keras.layers.Dense(256, activation=activation, dtype=dtype),
        tf.keras.layers.Dense(256, activation=activation, dtype=dtype),
        tf.keras.layers.Dense(512, activation=activation, dtype=dtype),
    ]

    layers.append(tf.keras.layers.Dense(output_shape, activation=activation, dtype=dtype))
    
    # To create a standard feedforward DNN, use Keras Sequential model
    model = tf.keras.models.Sequential(layers)
    return model

def defineLinearDecoder(output_shape, activation, dtype=tf.float64):
    layers = [
        tf.keras.layers.Dense(output_shape, activation=activation, dtype=dtype)
    ]
    model = tf.keras.models.Sequential(layers)
    return model

def plotResults(history, model, test_ds):
    fig, ax = plt.subplots()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='lower right')

    n_samples = 4
    batch = next(iter(test_ds))[0].numpy()[:n_samples]
    output = model(batch).numpy()
    side_length = int(batch.shape[1]**0.5)
    fig, ax = plt.subplots(n_samples, 2)

    for i in range(n_samples):
        ax[i, 0].imshow(batch[i, :].reshape((side_length, side_length)), vmin=0, vmax=1)
        ax[i, 1].imshow(output[i, :].reshape((side_length, side_length)), vmin=0, vmax=1)
    ax[0, 0].set_title("Original Data")
    ax[0, 1].set_title("Model output")
    return

def testResidualCompression(dataset, model, res_model):
    baseline_error = 0
    residual_comp_error = 0
    n_batches = 0
    for batch in dataset:
        decompressed = model(batch[0])
        residual = batch[0] - decompressed 
        decompressed_res = res_model(residual)
        final_residual = batch[0] - (decompressed + decompressed_res)
        baseline_error += tf.reduce_mean(tf.math.reduce_euclidean_norm(residual, axis=[1]) / tf.math.reduce_euclidean_norm(batch[0], axis=[1]))
        residual_comp_error += tf.reduce_mean(tf.math.reduce_euclidean_norm(final_residual, axis=[1]) / tf.math.reduce_euclidean_norm(batch[0], axis=[1]))
        n_batches += 1

    average_baseline_error = baseline_error / n_batches
    average_residual_comp_error = residual_comp_error / n_batches
    print(f"baselie:  {average_baseline_error}")
    print(f"residual: {average_residual_comp_error}")

def buildResidualDs(model):
    def f(item, label):
        return (item - model(item), item - model(item))
    return f
    
def printSummary(enc, dec, model, data):
    enc(next(iter(data))[0])
    dec(enc(next(iter(data))[0]))
    model(next(iter(data))[0])
    model.summary()
    return


if __name__ == '__main__':
    # hyperparameters
    activation = 'elu'
    batch_size = 128
    learning_rate = 1e-3
    n_epochs = 10
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    latent_space_dimension = 32

    train_filenames = [f'numpy/states_neumann_{n}.npy' for n in range(15)]
    test_filenames = [f'numpy/states_neumann_{n}.npy' for n in range(15, 20)]
    dirichlet_filenames = [f'numpy/states_dirichlet_{n}.npy' for n in range(2)]
    train_ds, train_info = loadDataset(train_filenames, batch_size)
    test_ds, test_info = loadDataset(test_filenames, batch_size)
    dirichlet_ds, dirichlet_info = loadDataset(dirichlet_filenames, batch_size)
    print(f"Training with {train_info['n_elements']} samples")
    
    # Define encoder and decoder models. A Keras Sequential model
    # can treat models as layers, so create full autoencoder model
    # by combining encoder and decoder. 
    encoder = defineEncoder(latent_space_dimension, activation)
    decoder = defineDecoder(train_info['element_shape'], activation)
    model = tf.keras.models.Sequential([encoder, decoder])
    res_encoder = defineEncoder(latent_space_dimension, activation)
    res_decoder = defineDecoder(train_info['element_shape'], activation)
    res_model = tf.keras.models.Sequential([res_encoder, res_decoder])
    printSummary(encoder, decoder, model, train_ds)
    printSummary(res_encoder, res_decoder, res_model, train_ds)
    
    # Tell TF what optimizer and loss to associate with model.
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
    )
    res_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(),
    )

    # Train the model.
    n_loops = 4
    for n in range(n_loops):
        print(f"\nStarting iteration {n+1}/{n_loops}")
        print("\tTraining main model")
        history = model.fit(train_ds, epochs=3, validation_data=test_ds)
        train_residual_ds = train_ds.map(buildResidualDs(model))
        test_residual_ds = test_ds.map(buildResidualDs(model))
        print("\n\tTraining residual autoencoder")
        res_history = res_model.fit(train_residual_ds, epochs=n_epochs, validation_data=test_residual_ds)
        optimizer.lr.assign(optimizer.lr / 5.0)
        print(optimizer.lr)
        
    # Plot results
    def composite_model(data):
        decompressed = model(data)
        residual = data - decompressed
        decompressed_res = res_model(residual)
        return decompressed + decompressed_res
    plotResults(history, composite_model, train_ds)

    # Test residual compression
    testResidualCompression(train_ds, model, res_model)
    residual_ds = train_ds.map(buildResidualDs(model))
    plotResults(res_history, res_model, residual_ds)
    plt.show()
