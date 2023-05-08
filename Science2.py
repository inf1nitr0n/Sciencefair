from re import X
from turtle import color
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from keras import Input, Model, callbacks, Sequential
import tensorflow as tf
import keras
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import warnings
import pydot
import xlsxwriter
import visualkeras
import uuid

warnings.filterwarnings("ignore", category=DeprecationWarning) 
print (tf.config.list_physical_devices('GPU'))


# Settings of the program 
class Settings():
    #directory odf samples
    samples_directory="./science/"

    directories=["samples05", "samples10","samples25","samples50","samples75"]

    dot_img_file1="./model1.png"
    dot_img_file2="./model2.png"
    dot_img_file3="./model3.png"
    batch_size=64

    workbook="./"+str(uuid.uuid4()) +".xlsx"
    result_treshold=0.5

    sizex=250
    sizey=500
    layers=3
    epochs=5
    learning_rate=2e-5
    run_eagerly=False


# read samples
class Samples():
    def convert(self, img):
        #x=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (thresh, x) = cv2.threshold(x, 150, 255, cv2.THRESH_BINARY)
        x=cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        #cv2.imwrite("./files/"+str(uuid.uuid4()) +".jpg", x)
        return x

    def __init__(self, dir):
        self.train_dir  = Settings.samples_directory+dir+"/train"
        self.test_dir   = Settings.samples_directory+dir+"/test"

        self.train_datagen = ImageDataGenerator(
              rescale=1./255,
              rotation_range=5,
              width_shift_range=0,
              height_shift_range=0,
              shear_range=0.1,
              zoom_range=0.2,
              #horizontal_flip=True,
              fill_mode='nearest'
              ,preprocessing_function=self.convert
              )

        self.test_datagen = ImageDataGenerator(
            rescale=1./255, 
            preprocessing_function=self.convert
            )

        self.train_generator = self.train_datagen.flow_from_directory(
                self.train_dir,
                target_size=(Settings.sizex,Settings.sizey),
                batch_size=Settings.batch_size,
                class_mode='binary')

        self.test_generator = self.test_datagen.flow_from_directory(
                self.test_dir,
                target_size=(Settings.sizex,Settings.sizey),
                batch_size=Settings.batch_size,
                class_mode='binary')

# base class of the neural network
class XModel(tf.keras.Model):
    def __init__(self, samples, results):
        super(XModel, self).__init__()
        self.samples=samples
        self.resuls=results
        self.history=None
    
    def show(self):
        acc = self.history.history["acc"]
        loss = self.history.history["loss"]
        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, "go", label="Accuracy" , color='b' )
        plt.plot(epochs, loss, "go", label="Loss", color = 'r')
        plt.title("Uspech v uceni")
        plt.legend()
        #plt.figure()
        plt.show()

    def build_and_fit(self):
        self.compile(
            loss="binary_crossentropy", 
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=Settings.learning_rate), 
            metrics=["acc"],
            run_eagerly=Settings.run_eagerly)

        earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True,
                                        verbose=1)

        self.history=self.fit_generator(
            self.samples.train_generator,
            steps_per_epoch=self.samples.train_generator.samples//self.samples.train_generator.batch_size+1,
            epochs=Settings.epochs,
            verbose=2)


    def write_results(self):
        print (self.evaluate(self.samples.test_generator))
        for x in range (0, len(self.samples.test_generator[0][0])):
            test_input = self.samples.test_generator[0][0][x]    
            test_input = np.expand_dims(test_input,axis=0)
            pred=self.predict_generator(test_input)
            print('Sample.: ', self.samples.test_generator._filepaths[x] , ' , from : ', len(self.samples.test_generator[0][0]), ' , result: ',  pred)

    def write_results_xls(self, dir):
        acc = self.history.history["acc"]
        loss = self.history.history["loss"]
        self.resuls.write_acc_loss(dir, acc, loss)

        for x in range (0, len(self.samples.test_generator[0][0])):
            test_input = self.samples.test_generator[0][0][x]    
            test_input = np.expand_dims(test_input,axis=0)
            pred=model.predict_generator(test_input)
            if (float(pred)<Settings.result_treshold):
                recognized=0
            else:
                recognized=1
            self.resuls.write(dir, self.samples.test_generator._filepaths[x], round(float(pred), 3), float(self.samples.test_generator.labels[x]), recognized)

    def plot_results(self):
        for x in range (0, len(self.samples.test_generator[0][0])):
            test_input = self.samples.test_generator[0][0][x]    
            test_input = np.expand_dims(test_input,axis=0)
            pred=model.predict_generator(test_input)
            plt.imshow(self.samples.test_generator[0][0][x])
            plt.title(pred)
        

# Neural network
class MyModel(XModel):
    def __init__(self, samples, results):
        super(MyModel, self).__init__(samples, results)
    
        self.cnn1 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(Settings.layers,Settings.sizex,Settings.sizey))
        self.mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.cnn2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.mp2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.cnn3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.mp3= tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1024, activation='relu')
        self.dense2 = tf.keras.layers.Dense(512, activation='relu')
        self.dense3 = tf.keras.layers.Dense(1, activation='sigmoid')
        
        #self.history=None

    def call(self, inputs):

        x = self.cnn1(inputs)
        x = self.mp1(x)
        x = self.cnn2(x)
        x = self.mp2(x)
        x = self.cnn3(x)
        x = self.mp3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

    def document(self):
        input_shape = (None, Settings.sizex,Settings.sizey, Settings.layers)
        self.build(input_shape)
        self.summary()
        tf.keras.utils.plot_model(self.build_graph(), to_file=Settings.dot_img_file1, dpi=300, show_shapes=True, show_layer_names=True, expand_nested=False )
        visualkeras.layered_view(self, to_file=Settings.dot_img_file2)

    def build_graph(self):
        x = Input(shape=(Settings.sizex, Settings.sizey, Settings.layers))
        return Model(inputs=[x], outputs=self.call(x))

    # results 
class Results():
    def __init__(self):
        self.workbook = xlsxwriter.Workbook(Settings.workbook)
        self.worksheet = self.workbook.add_worksheet('Results')
        self.worksheet.write(0, 0, 'sampleset')
        self.worksheet.write(0, 1, 'filename')
        self.worksheet.write(0, 2, 'value')
        self.worksheet.write(0, 3, 'expected')
        self.worksheet.write(0, 4, 'recognized')

        self.worksheet2 = self.workbook.add_worksheet('Accuracy')
        self.worksheet2.write(0, 0, 'sampleset')
        self.worksheet2.write(0, 1, 'acc')
        self.worksheet2.write(0, 2, 'loss')

        self.row=1
        self.row2start=1

    def write(self, sampleset, filename, value, expected, recognized):
        self.worksheet.write(self.row, 0, sampleset)
        self.worksheet.write(self.row, 1, filename)
        self.worksheet.write(self.row, 2, value)
        self.worksheet.write(self.row, 3, expected)
        self.worksheet.write(self.row, 4, recognized)
        self.row=self.row+1

    def write_acc_loss(self, sampleset, acc, loss):
        i=0;
        for x in acc:
            self.worksheet2.write(self.row2start+i,  0, sampleset)
            self.worksheet2.write(self.row2start+i,  1, round(x,3))
            i=i+1

        i=0;
        for x in loss:
            self.worksheet2.write(self.row2start+i, 2, round(x,3))
            i=i+1
        self.row2start=self.row2start+i

        
    def close(self):
        self.workbook.close()


# Main Program
settings=Settings()
results=Results()
for x in Settings.directories:
    samples=Samples(x)
    model = MyModel(samples, results)
    model.document()
    model.build_and_fit()
    model.write_results_xls(x)
    #model.write_results()
results.close()



