import glob
import matplotlib.pyplot as plt

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
import os

# Get count of number of files in this folder and all subfolders


def get_num_files(path):
    if not os.path.exists(path):
        return 0
    return sum([len(files) for r, d, files in os.walk(path)])

# Get count of number of subfolders directly below the folder in path


def get_num_subfolders(path):
    if not os.path.exists(path):
        return 0
    return sum([len(d) for r, d, files in os.walk(path)])

#   Define image generators that will variations of image with the image r/otated slightly, shifted up, down, left, or right,
#     sheared, zoomed in, or flipped horizontally on the vertical axis (ie. person looking to the left ends up looking to the right)


def create_img_generator():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )


width, height = 299, 299
epochs = 2
batch_size = 32
number_fc_neurons = 1024

train_dir = './data/train'
validate_dir = './data/validate'
num_train_samples = get_num_files(train_dir)
num_classes = get_num_subfolders(train_dir)
num_validate_samples = get_num_files(validate_dir)

train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

# Define data pre-processing
#   Define image generators for training and testing
train_image_gen = create_img_generator()
test_image_gen = create_img_generator()

#   Connect the image generator to a folder contains the source images the image generator alters.
#   Training image generator
train_generator = train_image_gen.flow_from_directory(
    train_dir,
    target_size=(width, height),
    batch_size=batch_size,
    seed=42  # set seed for reproducability
)

#   Validation image generator
validation_generator = test_image_gen.flow_from_directory(
    validate_dir,
    target_size=(width, height),
    batch_size=batch_size,
    seed=42  # set seed for reproducability
)

# include_top=False excludes the final FC layer
inception_v3_base_model = InceptionV3(weights='imagenet', include_top=False)
print('InceptionV2 base model without last FC loaded')


# define our own classifier
x = inception_v3_base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(number_fc_neurons, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# create new model which links inputs from v3 intou our predictions classifieation layers

model = Model(inputs=inception_v3_base_model.input, outputs=predictions)
print(model.summary())

# adding code not to re-train the old ones
# basic transfer learning
print('performing basic transfer learning')
# freeze all layers in inceptionv3 base
for layer in inception_v3_base_model.layers:
    layer.trainable = False

# regular things
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit_generator(train_generator,
                           epochs=epochs,
                           steps_per_epoch=num_train_samples,  # batch size
                           validation_data=validation_generator,
                           validation_steps=num_validate_samples,  # batch size
                           class_weight='auto')

# Save transfer learning model
model.save('inceptionv3-transfer-learning.model')
