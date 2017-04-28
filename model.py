# 
# This file contains the script to create and train the model
#

# Import section
import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Lambda, Cropping2D, Convolution2D, ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras import optimizers
import sklearn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define the model architecture, in this case use the NVIDIA end-to-end driving model with little modifications
def model_architecture():    
    # Define the model, we use Xavier initializer on each layer
    model = Sequential()    
    
    # Normalize data between [-1;1] and center around zero
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    
    # Five convolutional layer with dropout
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    # Flatten layer
    model.add(Flatten())
    
    # Three fully connected layer with dropout on first two
    model.add(Dense(100, init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init='glorot_uniform'))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(10, init='glorot_uniform'))
    model.add(ELU())
    
    # Output layer, predicts steering angle
    model.add(Dense(1, init='glorot_uniform'))
    
    return model

# Python generator to generate data for training on the fly, rather than storing everything in memory
def generator(samples, sample_size, mode):
    # Total number of samples
    num_samples = len(samples)
    # Different cameras available
    if mode == 'train':
        cameras = ['center', 'left', 'right']
    else:
        cameras = ['center']
        
    # Generator loop
    while True:
        # Shuffle data set for every batch
        sklearn.utils.shuffle(samples, random_state=43)
        
        # Go through one batch with batch_size steps
        for offset in range(0, num_samples, sample_size):
            # Get the samples for one batch
            batch_samples = samples[offset:offset+sample_size]
            
            # Store the images and angles
            images = []
            angles = []
                
            # Get the images for one batch
            for sample in batch_samples:
                # Get the different camera angles for every sample
                for cam in cameras:
                    # Choose randomly what kind of augmentation to use
                    if mode == 'train':
                        augmentation = np.random.choice(['flipping', 'brightness', 'shift', 'none'])
                    else:
                        augmentation = 'none'
                    
                    # Image and angle
                    image = None
                    angle = float(sample[3])
    
                    # Load the center image
                    if cam == 'center':
                        image = cv2.imread('./data/' + sample[0])
                    # Load the left image and alter angle
                    elif cam == 'left':
                        image = cv2.imread('./data/' + sample[1])
                        angle += 0.2
                    # Load the right image and alter angle
                    elif cam == 'right':
                        image = cv2.imread('./data/' + sample[2])
                        angle -= 0.2
                        
                    # Convert the image to RGB
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                        
                    # Flip the image and correct angle
                    if augmentation == 'flipping':
                        image = cv2.flip(image, 1)
                        angle *= -1.0
                    # Change the brightness of the image randomly
                    elif augmentation == 'brightness':
                        # Convert to HSV color space
                        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
                        # Extend range
                        image = np.array(image, dtype = np.float64)
                        # Choose new value randomly
                        brightness = np.random.uniform() + 0.5
                        # Alter value channel
                        image[:,:,2] = image[:,:,2] * brightness
                        # When value is above 255, set it to 255
                        image[:,:,2][image[:,:,2] > 255] = 255
                        # Convert back to 8-bit value
                        image = np.array(image, dtype = np.uint8)
                        # Convert back to RGB color space
                        image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
                        
                        # Apply a random shift to the image
                    elif augmentation == 'shift':
                        # Translation in x direction
                        trans_x = np.random.randint(0,100) - 50
                         # Correct angle
                        angle += trans_x * 0.004
                        # Translation in y direction
                        trans_y = np.random.randint(0,40)- 20
                        # Create the translation matrix
                        trans_matrix = np.float32([[1,0,trans_x],[0,1,trans_y]])
                        image = cv2.warpAffine(image,trans_matrix,(320, 160))
                        
                    # Crop the image
                    image = image[50:140, 0:320]
                    
                    # Resize to 200x66 pixel
                    image = cv2.resize(image, (200,66), interpolation=cv2.INTER_AREA)
                        
                    # Add image and angle to the list
                    images.append(np.reshape(image, (1, 66,200,3)))
                    angles.append(np.array([[angle]]))
                    
            # Return the next batch of samples shuffled
            X_train = np.vstack(images)
            y_train = np.vstack(angles)
            yield sklearn.utils.shuffle(X_train, y_train, random_state=21)
	
# Load the all the samples from the csv file and store them into single sample lines	
samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
print("Data sets loaded!")

# Hyper-parameter
epochs = 20
sample_size = 64
learning_rate = 1e-04

# Get training (80%) and validation (20%) sample lines
train_samples, valid_samples = train_test_split(samples, test_size=0.2)

# Init generators
train_generator = generator(train_samples, sample_size, 'train')
validation_generator = generator(valid_samples, sample_size, 'valid')

# Eventually load previous checkpoints in order not to start all over
#model.load_weights('./checkpoints/model-e000.h5')

# Show message that we start
print("Training network..")
print()

# Init the model
model = model_architecture()

# Loss and optimizer
model.compile(loss='mse', optimizer=optimizers.Adam(lr=learning_rate))

# Callbacks for checkpoints and early stop
check_point = ModelCheckpoint('./checkpoints/model-e{epoch:03d}.h5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto')
early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='min')

# Train the model
history = model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 3, validation_data=validation_generator,nb_val_samples=len(valid_samples), nb_epoch=epochs, verbose=1, callbacks=[early_stop, check_point])

# Save it to a file and show message again
model.save('model.h5')
print()
print("Network trained!")



