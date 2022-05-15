# -*- coding: utf-8 -*-
import os
import tensorflow as tf

import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
sess = tf.Session(config=config)

KTF.set_session(sess)
import data_helper
from siamese_NN import siamese_model
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from stats_graph import stats_graph
def train():
    model_path = './model/weights.best.hdf5'
    tensorboard_path = './model/ensembling'

    data = data_helper.load_pickle('model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_q3 = data['train_q3']
    train_q4 = data['train_q4']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_q3 = data['dev_q3']
    dev_q4 = data['dev_q4']
    dev_y = data['dev_label']
    
    model = siamese_model()
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    tensorboard = TensorBoard(log_dir=tensorboard_path)    
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max')
    callbackslist = [checkpoint, tensorboard,earlystopping,reduce_lr]

    model.fit([train_q1, train_q2, train_q3, train_q4], train_y,
              batch_size=512,
              epochs=200,
              validation_data=([dev_q1, dev_q2, dev_q3, dev_q4], dev_y),
              callbacks=callbackslist)

if __name__ == '__main__':
    train()