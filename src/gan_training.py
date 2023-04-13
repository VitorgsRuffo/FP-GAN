import sys

if len(sys.argv) <= 1:
    print("Please.\nSpecify the dataset:\n\t(orion1 | orion2)\n\tcic\n")
    quit(1)
dataset = sys.argv[1]


from import_data import import_orion_normal_data, import_cic_normal_data

# step 0: import data
if dataset == 'orion1':
    normal_day, scaler = import_orion_normal_data(dataset=1)

elif dataset == 'orion2':
    normal_day, scaler = import_orion_normal_data(dataset=2)

else:
    normal_day, scaler = import_cic_normal_data()



#setting random state for reproducible results.
from tensorflow.keras.utils import set_random_seed
set_random_seed(58)
#set_random_seed(45) ###change3


# step 1: build the generator and discriminator models.
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
import tensorflow as tf


#vanilla DNN  (CIC)
def build_cic_generator(latent_space_dim, no_of_features):
    model = Sequential()
    model.add(Dense(32, input_dim=latent_space_dim))
    model.add(LeakyReLU(0.2))

    model.add(Dense(32))
    model.add(LeakyReLU(0.2))

    model.add(Dense(no_of_features, activation='tanh'))
    return model


def build_cic_discriminator(no_of_features):
    model = Sequential()
    model.add(Dense(64, input_dim=no_of_features))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    return model


#vanilla DNN (ORION)
def build_orion_generator(latent_space_dim, no_of_features):
    model = Sequential()
    model.add(Dense(16, input_dim=latent_space_dim))
    model.add(LeakyReLU(0.2))

    model.add(Dense(16))
    model.add(LeakyReLU(0.2))

    model.add(Dense(no_of_features, activation='tanh'))
    return model


def build_orion_discriminator(no_of_features):
    model = Sequential()
    model.add(Dense(64, input_dim=no_of_features))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    
    model.add(Dense(64))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.2))
    
    model.add(Dense(1, activation='sigmoid'))
    return model



latent_space_dim = 128
no_of_features = 7 
#no_of_features = 6 ###change1
batch_size = None
generator = None
discriminator = None

if dataset == 'orion1' or dataset == 'orion2':
    batch_size = 256
    generator = build_orion_generator(latent_space_dim, no_of_features)
    discriminator = build_orion_discriminator(no_of_features)
else:
    batch_size = 32
    generator = build_cic_generator(latent_space_dim, no_of_features)
    discriminator = build_cic_discriminator(no_of_features)
   
generator.summary()
discriminator.summary()

# step 2: setting up training. (We'll redefine the fit method for making a custom training loop for GANs.)

# ps: There must be some balance between the learning of the generator and discriminator. One should not
# overperform the other, they must progress equally.

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd



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
        fake_data = self.generator(latent_space, training=False)
        
        # Train the discriminator
        with tf.GradientTape() as d_tape: 

            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator(real_data, training=True) 
            yhat_fake = self.discriminator(fake_data, training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)
            

            # Create labels for real and fakes images
            y_real = tf.zeros_like(yhat_real)
            y_fake = tf.ones_like(yhat_fake)
            y_realfake = tf.concat([y_real, y_fake], axis=0)
            

            # Add some noise to the TRUE outputs (in order to difficult discriminator learning) ##
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
epochs = 20

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
#         generated_normal_day = self.model.generator(latent_space, training=False) 


#         #generated_data = scaler.inverse_transform(generated_normal_day) #scaling data to original range

#         columns = \
#             ["timestamp", "bits", "dst_ip_entropy",  "dst_port_entropy", "src_ip_entropy", "src_port_entropy", "packets"]
        

#         data_frame = pd.DataFrame(generated_normal_day, columns=columns, index=None)
#         data_frame = data_frame.round({"timestamp":0}) #making sure the column values are integers.
        

#         # for each timestamp select 60 random samples (totalling 86400 samples)
#         reduced_data_frame = data_frame.groupby('timestamp', group_keys=False).apply(lambda x: x.sample(60) if x.shape[0] >= 60 else x.sample(frac=1))


#         reduced_data_frame.to_csv(f'./generated_data/generated_normal_day_epoch_{epoch}.csv', index=False)
#         os.system(f'python3 gen_data_visualization.py ./generated_data/generated_normal_day_epoch_{epoch}.csv 1')







###########################
#step 3: training GAN.
hist = gan.fit(normal_day, batch_size=batch_size, epochs=epochs)
#hist = gan.fit(normal_day, batch_size=batch_size, epochs=epochs, callbacks=[GanMonitor(latent_space_dim, scaler)])


import matplotlib as mpl

#axis format configuration:

import locale
plt.rcParams['axes.formatter.use_locale'] = True

plt.plot(hist.history['d_loss'], color='#379237', label='Erro do discriminator')
plt.plot(hist.history['g_loss'], color='#FF0303', label='Erro do generator')


locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
import matplotlib.ticker as tkr
def func(x, pos):  # formatter function takes tick label and tick position
    return locale.format_string("%.2f", x)
axis_format = tkr.FuncFormatter(func)  # make formatter

mpl.rcParams['lines.linewidth'] = 1

plt.xlabel('Época')
plt.ylabel('Erro')
ax = plt.gca()

from matplotlib.ticker import MaxNLocator
ax.xaxis.set_major_locator(MaxNLocator(integer=True)) 
ax.yaxis.set_major_formatter(axis_format) #when using ','

import numpy as np
plt.yticks(np.arange(0.50, 0.90, step=0.05))

plt.legend(loc='upper right')
plt.margins(x=0)
plt.savefig(f"./gan_loss.png", dpi=150)
plt.close()


#step 4: save models
generator.save('./model/generator')
discriminator.save('./model/discriminator')



# step 5: check learned distribution: ###change2

#generator = load_model('./model/generator', compile=False)

amount_of_samples_to_generate = None
if dataset == 'orion1' or dataset == 'orion2':
    amount_of_samples_to_generate = 1000000 #it will generate the amount-of-seconds-in-a-day samples
else:
    amount_of_samples_to_generate = 3718 #it will generate the amount-of-seconds-in-a-day samples

latent_space = tf.random.normal((amount_of_samples_to_generate, latent_space_dim)) 
generated_normal_day = generator(latent_space, training=False) 
generated_normal_day = scaler.inverse_transform(generated_normal_day) #scaling data to original range


columns = ["timestamp", "bits", "dst_ip_entropy",  "dst_port_entropy", "src_ip_entropy", "src_port_entropy", "packets"]
# ["bits", "dst_ip_entropy",  "dst_port_entropy", "src_ip_entropy", "src_port_entropy", "packets"]

data_frame = pd.DataFrame(generated_normal_day, columns=columns, index=None)

data_frame = data_frame.round({"timestamp":0}) #making sure the column values are integers.

# for each timestamp select one random sample (totalling 86400 samples)
data_frame = data_frame.groupby('timestamp', group_keys=False).apply(lambda x: x.sample(1))
indices = data_frame['timestamp'].isin(range(0,86400))
data_frame = data_frame[indices]



data_frame.to_csv(f'./generated_data/generated_normal_day.csv', index=False)
os.system(f'python3 gen_data_visualization.py ./generated_data/generated_normal_day.csv {dataset}')
