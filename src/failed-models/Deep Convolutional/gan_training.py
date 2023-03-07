from import_data import import_orion_normal_data


# step 0: import data
regular_day, scaler = import_orion_normal_data()


# step 1: build the generator and discriminator models.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, LeakyReLU, ReLU, Dropout, BatchNormalization
from tensorflow.keras import initializers, Input
import tensorflow as tf


#vanilla DNN
# def build_generator(latent_space_dim):
#     model = Sequential()
#     model.add(Dense(32, input_dim=latent_space_dim))
#     model.add(LeakyReLU(0.2))
#     model.add(Dense(32))
#     model.add(LeakyReLU(0.2))
#     model.add(Dense(7, activation='tanh'))
#     return model


# def build_discriminator():
#     model = Sequential()
#     model.add(Dense(64, input_dim=7))
#     model.add(LeakyReLU(0.2))
#     model.add(Dropout(0.2))
    
#     model.add(Dense(64))
#     model.add(LeakyReLU(0.2))
#     model.add(Dropout(0.2))

#     model.add(Dense(64))
#     model.add(LeakyReLU(0.2))
#     model.add(Dropout(0.2))
    
#     model.add(Dense(1, activation='sigmoid'))
#     return model


#vanilla DNN 2
# def build_generator(latent_space_dim):
#     model = Sequential()
#     model.add(Dense(8, input_dim=latent_space_dim))
#     model.add(LeakyReLU(0.2))
#     model.add(Dense(8))
#     model.add(LeakyReLU(0.2))
#     model.add(Dense(7, activation='tanh'))
#     return model


# def build_discriminator():
#     model = Sequential()
#     model.add(Dense(32, input_dim=7))
#     model.add(LeakyReLU(0.2))
#     model.add(Dropout(0.2))
    
#     model.add(Dense(32))
#     model.add(LeakyReLU(0.2))
#     model.add(Dropout(0.2))
    
#     model.add(Dense(1, activation='sigmoid'))
#     return model









# Deep Convolutional GAN 1.
# def build_generator(latent_space_dim):
#     model = Sequential()
    
#     model.add(Conv1D(32, kernel_size=1, input_shape=(1, latent_space_dim)))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     #print(model.output_shape)

#     model.add(Conv1D(16, kernel_size=1))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU())
#     #print(model.output_shape)

#     model.add(Conv1D(7, kernel_size=1, activation='tanh'))
#     #print(model.output_shape)

#     #print(model.summary())

#     return model


# def build_discriminator():
#     model = Sequential()

#     model.add(Conv1D(64, kernel_size=1, input_shape=(1, 7)))
#     model.add(LeakyReLU())
#     model.add(Dropout(0.3))
#     #print(model.output_shape)

#     model.add(Conv1D(64, kernel_size=1))
#     model.add(LeakyReLU())
#     model.add(Dropout(0.3))
#     #print(model.output_shape)

#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))

#     #print(model.summary())

#     return model



# Deep Convolutional GAN 2 
def build_generator(latent_space_dim):
    model = Sequential()
    
    model.add(Conv1D(8, kernel_size=1, input_shape=(1, latent_space_dim)))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #print(model.output_shape)

    model.add(Conv1D(8, kernel_size=1))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    #print(model.output_shape)

    model.add(Conv1D(7, kernel_size=1, activation='tanh'))
    #print(model.output_shape)

    #print(model.summary())

    return model


def build_discriminator():
    model = Sequential()

    model.add(Conv1D(32, kernel_size=1, input_shape=(1, 7)))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    #print(model.output_shape)

    model.add(Conv1D(32, kernel_size=1))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))

    #print(model.output_shape)

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    #print(model.summary())

    return model



# Deep Convolutional GAN 3.
# Implement gen and disc with shape (features, 1)
    #   - Convergence failure: disc is always learning. gen doesnt learn. or vice versa.
# def build_generator(latent_space_dim):
#     model = Sequential()
#     #initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
#     model.add(Conv1D(16, kernel_size=32, input_shape=(latent_space_dim, 1)))#, kernel_initializer=initializer))
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     #model.add(Dropout(0.5))
#     #print(model.output_shape)

#     model.add(Conv1D(16, kernel_size=16))#, kernel_initializer=initializer))
#     model.add(BatchNormalization())
#     model.add(ReLU())
#     #model.add(Dropout(0.5))

#     #print(model.output_shape)

#     model.add(Conv1D(1, kernel_size=12, activation='tanh'))#, kernel_initializer=initializer))
#     #print(model.output_shape)

#     print(model.summary())

#     return model



# def build_discriminator():
#     model = Sequential()
#     #initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
#     model.add(Conv1D(32, kernel_size=2, input_shape=(7, 1)))#, kernel_initializer=initializer))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.5))
#     #print(model.output_shape)

#     model.add(Conv1D(32, kernel_size=2))#, kernel_initializer=initializer))
#     model.add(BatchNormalization())
#     model.add(LeakyReLU(alpha=0.3))
#     model.add(Dropout(0.5))
#     #print(model.output_shape)


#     model.add(Flatten())
#     model.add(Dense(1, activation='sigmoid'))#, kernel_initializer=initializer))

#     print(model.summary())

#     return model



batch_size = 256
latent_space_dim = 128
#latent_space_dim = 16 #DCGAN change (2) 

generator = build_generator(latent_space_dim)
generator.summary()

discriminator = build_discriminator()
discriminator.summary()


# step 2: setting up training. (We'll redefine the fit method for making a custom training loop for GANs.)

# ps: There must be some balance between the learning of the generator and discriminator. One should not
# overperform the other, they must progress equally.

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import random



class OrionGAN(Model): 
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator 
        self.discriminator = discriminator 
        
    def compile(self, g_opt, d_opt, g_loss, d_loss, batch_size, latent_space_dim, *args, **kwargs): 
        super().compile(*args, **kwargs)
        self.g_opt = g_opt
        self.d_opt = d_opt
        self.g_loss = g_loss
        self.d_loss = d_loss 
        self.batch_size = batch_size
        self.latent_space_dim = latent_space_dim


    def train_step(self, batch): #called inside fit method.
        real_data = batch

        latent_space = tf.random.normal((self.batch_size, self.latent_space_dim))
        latent_space = tf.reshape(latent_space, [self.batch_size, 1, self.latent_space_dim]) ## DCGAN change
        #latent_space = tf.reshape(latent_space, [self.batch_size, self.latent_space_dim, 1]) ## DCGAN change

        fake_data = self.generator(latent_space, training=False)
        
        # Train the discriminator
        with tf.GradientTape() as d_tape: 

            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_data, training=True) 
            yhat_fake = self.discriminator(fake_data, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            

            # Create labels for real and fakes images
            y_real = tf.zeros_like(yhat_real)


            
            #randomly flipping real labels (in order to difficult discriminator learning)
            # indices = [random.randint(0, 255) for i in range(0, 256//8)]
            # y_real_np = np.zeros((self.batch_size))
            # y_real_np[indices] = 1
            # y_real_list = list(y_real_np)
            # y_real_tensor = tf.convert_to_tensor(y_real_list)
            # y_real_tensor = tf.reshape(y_real_tensor, [self.batch_size, 1])
            # y_real += y_real_tensor
          

       

            y_fake = tf.ones_like(yhat_fake)
            y_realfake = tf.concat([y_real, y_fake], axis=0)
            

            # Add some noise to the TRUE outputs (in order to difficult discriminator learning)
            # noise_real = 0.1*tf.random.uniform(tf.shape(yhat_real))
            # noise_fake = -0.1*tf.random.uniform(tf.shape(yhat_fake))
            # y_realfake += tf.concat([noise_real, noise_fake], axis=0)
            
            # Calculate loss - BINARYCROSS 
            total_d_loss = self.d_loss(y_realfake, yhat_realfake)
            
        # Apply backpropagation - nn learn 
        dgrad = d_tape.gradient(total_d_loss, self.discriminator.trainable_variables) 
        self.d_opt.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))
        
        # Train the generator 
        with tf.GradientTape() as g_tape: 
            # Generate some new data
            latent_space = tf.random.normal((self.batch_size, self.latent_space_dim))
            latent_space = tf.reshape(latent_space, [self.batch_size, 1, self.latent_space_dim]) ## DCGAN change
            #latent_space = tf.reshape(latent_space, [self.batch_size, self.latent_space_dim, 1]) ## DCGAN change

            fake_data = self.generator(latent_space, training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator(fake_data, training=False)
                                        
            # Calculate loss - trick to training to fake out the discriminator
            total_g_loss = self.g_loss(tf.zeros_like(predicted_labels), predicted_labels) 
            
        # Apply backprop
        ggrad = g_tape.gradient(total_g_loss, self.generator.trainable_variables)
        self.g_opt.apply_gradients(zip(ggrad, self.generator.trainable_variables))
        
        return {"d_loss":total_d_loss, "g_loss":total_g_loss}


g_opt = Adam(learning_rate=0.0002)
d_opt = Adam(learning_rate=0.00002)
#g_opt = Adam(learning_rate=0.0001)
#d_opt = Adam(learning_rate=0.00001) #generator is gonna learning faster than discriminator cause its task is harder
g_loss = BinaryCrossentropy()
d_loss = BinaryCrossentropy()
epochs = 100

gan = OrionGAN(generator, discriminator)
gan.compile(g_opt, d_opt, g_loss, d_loss, batch_size, latent_space_dim)

import os
from tensorflow.keras.callbacks import Callback

# class GanMonitor(Callback):
#     def __init__(self, latent_space_dim, scaler):
#         self.latent_space_dim = latent_space_dim
#         self.scaler = scaler


#     def on_epoch_end(self, epoch, logs=None):
#         amount_of_samples_to_generate = 864000 #it will generate 10x the amount-of-seconds-in-a-day samples
        
#         latent_space = tf.random.normal((amount_of_samples_to_generate, self.latent_space_dim)) 
#         generated_regular_day = self.model.generator(latent_space, training=False) 


#         #generated_data = scaler.inverse_transform(generated_regular_day) #scaling data to original range

#         columns = \
#             ["timestamp", "bytes", "dst_ip_entropy",  "dst_port_entropy", "src_ip_entropy", "src_port_entropy", "packets"]
        

#         data_frame = pd.DataFrame(generated_regular_day, columns=columns, index=None)
#         data_frame = data_frame.round({"timestamp":0}) #making sure the column values are integers.
        

#         # for each timestamp select 60 random samples (totalling 86400 samples)
#         reduced_data_frame = data_frame.groupby('timestamp', group_keys=False).apply(lambda x: x.sample(60) if x.shape[0] >= 60 else x.sample(frac=1))


#         reduced_data_frame.to_csv(f'./generated_data/generated_regular_day_epoch_{epoch}.csv', index=False)
#         os.system(f'python3 gen_data_visualization.py ./generated_data/generated_regular_day_epoch_{epoch}.csv 1')







###########################
# step 3: training GAN.
hist = gan.fit(regular_day, batch_size=batch_size, epochs=epochs)
#hist = gan.fit(regular_day, batch_size=batch_size, epochs=epochs, callbacks=[GanMonitor(latent_space_dim, scaler)])

plt.suptitle('Loss')
plt.plot(hist.history['d_loss'], label='d_loss')
plt.plot(hist.history['g_loss'], label='g_loss')
plt.legend()
plt.savefig(f"./gan_loss.png", dpi=150)
plt.close()


# step 4: save models
generator.save('./model/generator.h5')
discriminator.save('./model/discriminator.h5')

# step 5: check learned distribution:

#generator = load_model('./model/generator.h5', compile=False)


amount_of_samples_to_generate = 86400 #it will generate 10x the amount-of-seconds-in-a-day samples

latent_space = tf.random.normal((amount_of_samples_to_generate, latent_space_dim)) 
latent_space = tf.reshape(latent_space, [amount_of_samples_to_generate, 1, latent_space_dim]) ## DCGAN change
#latent_space = tf.reshape(latent_space, [amount_of_samples_to_generate, latent_space_dim, 1]) ## DCGAN change

generated_regular_day = generator(latent_space, training=False) 
generated_regular_day = np.reshape(generated_regular_day, (generated_regular_day.shape[0], generated_regular_day.shape[2])) ## DCGAN change
#generated_regular_day = np.reshape(generated_regular_day, (generated_regular_day.shape[0], generated_regular_day.shape[1])) ## DCGAN change

generated_regular_day = scaler.inverse_transform(generated_regular_day) #scaling data to original range


columns = \
    ["timestamp", "bytes", "dst_ip_entropy",  "dst_port_entropy", "src_ip_entropy", "src_port_entropy", "packets"]


data_frame = pd.DataFrame(generated_regular_day, columns=columns, index=None)
data_frame = data_frame.round({"timestamp":0}) #making sure the column values are integers.


# for each timestamp select one random sample (totalling 86400 samples)
# data_frame = data_frame.groupby('timestamp', group_keys=False).apply(lambda x: x.sample(1))



data_frame.to_csv(f'./generated_data/generated_regular_day.csv', index=False)
os.system(f'python3 gen_data_visualization.py ./generated_data/generated_regular_day.csv 1')
