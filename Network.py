import tensorflow as tf
import numpy as np
import librosa
import random
import os
import IPython.display
from ipywidgets import interact, interactive, fixed
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter
import scipy.ndimage
import time
import math
from random import shuffle

#--------------------------------------------------------------------------------------------------

#   Hyperparameters 

fmaps_gen = 60  # Number of filters in first layer of generator
fmaps_dis = 60  # Number of filters in first layer of discriminator
pool_size = 50  # pool_size
img_depth = 1   # RGB format
stddev = 0.02   # standard deviation
learning_rate = 0.0001  # Learning rate
path_A = ('/home/kraev_simeon/dataset/Classical/')  # Path to set A  
path_B = ('/home/kraev_simeon/dataset/rock/')       # Path to set B
spec_batch = 1     # Batch size
spec_length = 20   # length of spectrograms, in seconds
epochs = 20        # Number of epochs to train
to_restore = False # Restore the weights from a saved file
#-----------------------------------------------------------------------------------------------
### STFT parameters ###
fft_size = 2048            # window size for the FFT
step_size = fft_size / 16  # distance to slide along the window (in time)
spec_thresh = 4  # threshold for spectrograms (lower filters out more noise)
lowcut = 500     # Hz # Low cut for our butter bandpass filter
highcut = 15000  # Hz # High cut for our butter bandpass filter
# For mels
n_mel_freq_components = 64  # number of mel frequency channels
shorten_factor = 10         # how much should we compress the x-axis (time)
start_freq = 300            # Hz # What frequency to start sampling our melS from 
end_freq = 8000             # Hz # What frequency to stop sampling our melS from

#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------


def instance_norm(x):
    
    # Instance normalization
    
    with tf.variable_scope("instance_norm"):
        epsilon = 1e-9
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
        scale = tf.get_variable('scale',[x.get_shape()[-1]], 
            initializer=tf.truncated_normal_initializer(mean=1.0, stddev=0.03))
        offset = tf.get_variable('offset',[x.get_shape()[-1]],initializer=tf.constant_initializer(0.0))
        out = scale*tf.div(x-mean, tf.sqrt(var+epsilon)) + offset

        return out

def general_conv2d(inputconv, feature_maps, kernel_size, stride,name ,do_norm= True, do_relu=True, padding = "SAME" ):
    
    # Convolutional Layer
    
    with tf.variable_scope(name):
        
        conv = tf.layers.conv2d(inputconv, feature_maps, kernel_size, stride, padding ,activation= None,
                                use_bias = True,
                                kernel_initializer = tf.truncated_normal_initializer(stddev=stddev),
                                bias_initializer = tf.constant_initializer(0.0))
        if do_norm:
            conv = tf.contrib.layers.instance_norm(conv,center=True,scale=True,epsilon=1e-08)
         
        if do_relu:
            conv = tf.nn.leaky_relu(conv, 0.5, "lrelu")

        return conv        
        
    return conv 
        

def general_deconv2d(imp, feature_maps, kernel_height, kernel_width,stride_height, stride_width, name,
                     do_relu=True, padding = 'SAME'):
    
    # Deconvolutional Layer
    
    with tf.variable_scope(name):
        
        deconv = tf.contrib.layers.conv2d_transpose(imp, feature_maps, kernel_width, stride_height,padding="SAME", activation_fn=None,
                                                    weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                                    biases_initializer=tf.constant_initializer(0.0))
                                                   
        if do_relu:
            deconv = tf.nn.leaky_relu(deconv, 0.5, "lrelu")

        return deconv   
    
    return deconv
 
    
def resnet_blocks(input_res, feature_maps, name):
    
    # Residual Block
    
    with tf.variable_scope(name):
        
        out_res_1 = general_conv2d(input_res, feature_maps,(3,3),(1,1), name='out_res_1')
        out_res_2 = general_conv2d(out_res_1, feature_maps,(3,3),(1,1), name='out_res_2', do_relu=False)
        many_maps = tf.concat([ out_res_2, input_res ], 3 )
    return many_maps


def build_generator(input_gen, name):
    
    # Create the Generator
    
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        o_c1 = general_conv2d(input_gen, 95, (3, 3), (1, 1),name='o_c1')
        o_c2 = general_conv2d(o_c1, fmaps_gen*4, (3, 3),( 2, 2),name='o_c2')
        o_enc_A = general_conv2d(o_c2, fmaps_gen*5,( 3, 3), (2, 2),name='o_enc_A',do_norm = False, do_relu=False)

    # Residual Blocks 
        o_r1 = resnet_blocks(o_enc_A, feature_maps=40*4,name='o_r1')
        o_r2 = resnet_blocks(o_r1, feature_maps=40*4,name='o_r2')
        o_r3 = resnet_blocks(o_r2, feature_maps=40*4,name='o_r3')
        o_r4 = resnet_blocks(o_r3, feature_maps=40*4,name='o_r4')
        o_r5 = resnet_blocks(o_r4, feature_maps=40*4,name='o_r5')
        o_r6 = resnet_blocks(o_r5, feature_maps=40*4,name='o_r6')
        o_r7 = resnet_blocks(o_r6, feature_maps=40*4,name='o_r7')
        o_r8 = resnet_blocks(o_r7, feature_maps=40*4,name='o_r8')
        o_enc_B = resnet_blocks(o_r8, feature_maps=40*4,name='o_enc_B')
        o_enc_B = tf.concat([ o_enc_B, o_enc_A ], 3 )

    #Decoding
        o_d1 = general_deconv2d(o_enc_B, fmaps_gen*5, 3, 3, 2, 2, name = 'o_d1')
        o_d2 = general_deconv2d(o_d1, fmaps_gen*4, 3, 3, 1, 1, name = 'o_d2')
        o_d3 = general_deconv2d(o_d2, fmaps_gen*4, 3, 3, 2, 2, name = 'o_d3')
        o_d4 = general_deconv2d(o_d3, fmaps_gen*3, 3, 3, 1, 1, name = 'o_d4')
        gen_B = general_conv2d(o_d4, 1,( 3, 3), (1, 1),name = 'gen_B',do_norm = False, do_relu=False)
        
    return gen_B



def build_discriminator(inputdisc, name):
    
    # Create the Discriminator
    
    with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
        f = 3 # kernel dims
        
        patch_input = tf.random_crop(inputdisc,[1,16,16,1])
        d_c1 = general_conv2d(patch_input, fmaps_dis,( f, f), (2, 2), name = 'd_c1',do_norm = False)
        d_c2 = general_conv2d(d_c1, fmaps_dis*2,( f, f),( 2, 2), name = 'd_c2')#,do_norm = False)
        d_c3 = general_conv2d(d_c2, fmaps_dis*4, (f, f),( 2, 2), name = 'd_c3')#,do_norm = False)
        d_c4 = general_conv2d(d_c3, fmaps_dis*5, (f, f), (1, 1), name = 'd_c4')#,do_norm = False)
        d_c5 = general_conv2d(d_c4, 1, (f, f), (1, 1), name = 'd_c5', do_norm = False, do_relu=False)
            
    return d_c5



#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------

# Input images

input_A = tf.placeholder(tf.float32, [spec_batch, None, None, img_depth], name="input_A")
input_B = tf.placeholder(tf.float32, [spec_batch, None, None, img_depth], name="input_B")

# A pool of fake images 

fake_pool_A = tf.placeholder(tf.float32, [spec_batch, None, None, img_depth], name="fake_pool_A")
fake_pool_B = tf.placeholder(tf.float32, [spec_batch, None, None, img_depth], name="fake_pool_B")

# Input image dimentions

img_height = 64
img_width = 700

# Fakes images

fake_images_A = np.zeros(( pool_size, 1, img_height, img_width, img_depth ), dtype=np.float32 )
fake_images_B = np.zeros(( pool_size, 1, img_height, img_width, img_depth ), dtype=np.float32 )

num_fake_inputs = 0


def my_network(input_A, input_B):
    with tf.variable_scope('my_network') as scope:
        
        # Create the network - 4 Generators, 6 Discriminators
        
        gen_B = build_generator(input_A, name='gB') #generator_A2B
        gen_A = build_generator(input_B, name='gA') #generator_B2A
        dec_A = build_discriminator(input_A, name='dA')  #discriminator_A
        dec_B = build_discriminator(input_B, name='dB') #discriminator_B
        
        scope.reuse_variables()
#----------------------------------------------------------------------------------------        
        dec_gen_A = build_discriminator(gen_A, 'dA') #discriminator_B2A
        dec_gen_B = build_discriminator(gen_B, 'dB')  #discriminator_A2B
        cyc_A = build_generator(gen_B, 'gB') #generator_ABA
        cyc_B = build_generator(gen_A, 'gA') #generator_BAB
        
        scope.reuse_variables()
#---------------------------------------------------------------------------------------
        fake_pool_rec_A = build_discriminator(fake_pool_A,'dA') #discriminator_Fake_pool_rec_A
        fake_pool_rec_B = build_discriminator(fake_pool_B,'dB') #discriminator_Fake_pool_rec_B
        
    return gen_A, gen_B, dec_A, dec_B, dec_gen_A, dec_gen_B, cyc_A, cyc_B, fake_pool_rec_A, fake_pool_rec_B


# Creates a pool of 50 fake images and takes out a random image from comparison if a random number is above 0.5
            
def fake_image_pool(num_fakes, gen, fake_pool):
    
    if(num_fakes < pool_size):
        fake_pool[num_fakes] = gen
        return gen
    else:
        p = random.random()
        if p > 0.5:
            random_id = random.randint(0,pool_size-1)
            temp = fake_pool[random_id]
            fake_pool[random_id] = gen
            return temp
        else:
            return gen

gen_A, gen_B, dec_A, dec_B, dec_gen_A, dec_gen_B, cyc_A, cyc_B, fake_pool_rec_A, fake_pool_rec_B = my_network(input_A, input_B)


# Creates a list of all 

def Path_List(path):
    path_list = []
    for root,dirs,files in os.walk(path):
        for file_ in files:
            file_full = os.path.join(root, file_)
            path_list.append(file_full)
        return path_list
    
# Initializes the Mel filters needed to create the MEL spectrograms 
    
mel_filter, mel_inversion_filter = create_mel_filter(fft_size = 2048,
                                                        n_freq_components = n_mel_freq_components,
                                                        start_freq = start_freq,
                                                        end_freq = end_freq)

# Creates as many spectrograms from a song, given there is enough length. spec_length is the parameter deciding that. 

def SpecMaker(path_list, spec_length, index, offset):
    length = librosa.core.get_duration(filename = path_list[index])

    if length - offset >= spec_length :
        
        data,rate = librosa.load( path_list[index], sr=44100, duration = spec_length, offset = offset )
        offset += spec_length
        wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size = fft_size,step_size = step_size,
                                                                                 log = True, thresh = spec_thresh)

        wav_spectrogram = make_mel(wav_spectrogram, mel_filter, shorten_factor = 9.8)
        

        wav_spectrogram = np.expand_dims(wav_spectrogram, axis = 0)
        wav_spectrogram = np.expand_dims(wav_spectrogram, axis = 3)

    elif length - offset < spec_length  :
        path_list.pop(index)
        offset = 0
        length = librosa.core.get_duration(filename = path_list[index])
        data,rate = librosa.load( path_list[index], sr=44100, duration = spec_length, offset = offset )
        offset += spec_length
        wav_spectrogram = pretty_spectrogram(data.astype('float64'), fft_size = fft_size,step_size = step_size,
                                                                                 log = True, thresh = spec_thresh)

        wav_spectrogram = make_mel(wav_spectrogram, mel_filter, shorten_factor = 9.8)

        wav_spectrogram = np.expand_dims(wav_spectrogram, axis = 0)
        wav_spectrogram = np.expand_dims(wav_spectrogram, axis = 3)
        
    return wav_spectrogram, path_list, offset

# Cycle loss

cyc_loss = tf.reduce_mean(tf.abs(input_A-cyc_A)) + tf.reduce_mean(tf.abs(input_B-cyc_B))

# Loss for gen_A and gen_B

disc_loss_A = tf.reduce_mean(tf.squared_difference(dec_gen_A,1))
disc_loss_B = tf.reduce_mean(tf.squared_difference(dec_gen_B,1))

# Generator loss        

g_loss_A = cyc_loss*14 + disc_loss_B
g_loss_B = cyc_loss*14 + disc_loss_A

# Discriminator loss

d_loss_A = (tf.reduce_mean(tf.square(fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(dec_A,1)))/2.0
d_loss_B = (tf.reduce_mean(tf.square(fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(dec_B,1)))/2.0

# Using Adam for gradient descent 
optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2 = 0.999, epsilon=1e-08)

# Initialize the training variables

with tf.variable_scope("trainable_variables"):
    trainable_variables = tf.trainable_variables()
    d_A_vars = [var for var in trainable_variables if 'dA' in var.name]
    g_A_vars = [var for var in trainable_variables if 'gA' in var.name]
    d_B_vars = [var for var in trainable_variables if 'dB' in var.name]
    g_B_vars = [var for var in trainable_variables if 'gB' in var.name]
         
        
d_A_loss_summ = tf.summary.scalar("Discriminator_loss_A", d_loss_A)
d_B_loss_summ = tf.summary.scalar("Discriminator_loss_B", d_loss_B)
g_A_loss_summ = tf.summary.scalar("Generator_loss_A", g_loss_A)
g_B_loss_summ = tf.summary.scalar("Generator_loss_B", g_loss_B)
    
    # Minimize the loss
    
d_A_trainer = optimizer.minimize(d_loss_A, var_list = d_A_vars)
d_B_trainer = optimizer.minimize(d_loss_B, var_list = d_B_vars)
g_A_trainer = optimizer.minimize(g_loss_A, var_list = g_A_vars)
g_B_trainer = optimizer.minimize(g_loss_B, var_list = g_B_vars)

# Create summaries

summaries = tf.summary.merge_all()

# Save weights 
            
train_writer = tf.summary.FileWriter("./summaries/summaries_long_train/", tf.get_default_graph())