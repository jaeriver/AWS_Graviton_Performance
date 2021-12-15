import os, re, time, json
import PIL.Image, PIL.ImageFont, PIL.ImageDraw
import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
import sys
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--batch_list',
                      nargs='+',
                      required=True)

model_type = 'resnet50'
batch_list = parser.parse_args().batch_list

def deserialize_image_record(record):
    feature_map = {'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
                  'image/class/label': tf.io.FixedLenFeature([1], tf.int64, -1)}
    obj = tf.io.parse_single_example(serialized=record, features=feature_map)
    imgdata = obj['image/encoded']
    label = tf.cast(obj['image/class/label'], tf.int32)   
    return imgdata, label

def val_preprocessing(record):
    imgdata, label = deserialize_image_record(record)
    label -= 1
    
    image = tf.io.decode_jpeg(imgdata, channels=3, 
                              fancy_upscaling=False, 
                              dct_method='INTEGER_FAST')

    shape = tf.shape(image)
    height = tf.cast(shape[0], tf.float32)
    width = tf.cast(shape[1], tf.float32)
    side = tf.cast(tf.convert_to_tensor(256, dtype=tf.int32), tf.float32)

    scale = tf.cond(tf.greater(height, width),
                  lambda: side / width,
                  lambda: side / height)
    
    new_height = tf.cast(tf.math.rint(height * scale), tf.int32)
    new_width = tf.cast(tf.math.rint(width * scale), tf.int32)
    
    image = tf.image.resize(image, [new_height, new_width], method='bicubic')
    if model_type =='inception_resnet_v2':
        image = tf.image.resize_with_crop_or_pad(image, 299, 299)
    else:
        image = tf.image.resize_with_crop_or_pad(image, 224, 224)

    label = tf.cast(label, tf.int32)
    
    image = tf.cast(image, tf.float32)
    return image, label

def get_dataset(batch_size, use_cache=False):
    data_dir = '/home/ubuntu/datasets/images-50000/*'
    files = tf.io.gfile.glob(os.path.join(data_dir))
    dataset = tf.data.TFRecordDataset(files)
    
    dataset = dataset.map(map_func=val_preprocessing, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat(count=1)
    
    return dataset
        
def inference(model_type, batch_size):
    walltime_start = time.time()
    first_iter_time = 0
    iter_times = []
    pred_labels = []
    actual_labels = []
    total_datas = 50000
    display_every = 1000
    display_threshold = display_every
    
    ds = get_dataset(batch_size)
    saved_model_name = f"../../model/{model_type}"
    load_start = time.time()
    model = load_model(saved_model_name)
    load_time = time.time() - load_start
    counter = 0
    for batch, batch_labels in ds:
        start_time = time.time()
        yhat_np = model.predict(batch)
        if counter ==0:
            first_iter_time = time.time() - start_time
        else:
            iter_times.append(time.time() - start_time)
        actual_labels.extend(label for label_list in batch_labels for label in label_list)
        pred_labels.extend(list(np.argmax(yhat_np, axis=1)))

        if counter*batch_size >= display_threshold:
            print(f'Images {counter*batch_size}/{total_datas}. Average i/s {np.mean(batch_size/np.array(iter_times[-display_every:]))}')
            display_threshold+=display_every

        counter+=1
    iter_times = np.array(iter_times)
    acc = np.sum(np.array(actual_labels) == np.array(pred_labels))/len(actual_labels)
    results = pd.DataFrame(columns = [f'graviton_{model_type}_{batch_size}'])
    results.loc['batch_size']              = [batch_size]
    results.loc['accuracy']                = [acc]
    results.loc['first_prediction_time']   = [first_iter_time]
    results.loc['average_prediction_time'] = [np.mean(iter_times)]
    results.loc['load_time']               = [load_time]
    results.loc['wall_time']               = [time.time() - walltime_start]

    return results, iter_times


results = pd.DataFrame()

for batch_size in batch_list:
  opt = {'batch_size': batch_size}
  iter_ds = pd.DataFrame()
  
  print(f'{batch_size} start')
  res, iter_times = inference(model_type, int(batch_size))
  col_name = lambda opt: f'graviton_{model_type}_{batch_size}'
  
  iter_ds = pd.concat([iter_ds, pd.DataFrame(iter_times, columns=[col_name(opt)])], axis=1)
  results = pd.concat([results, res], axis=1)
  print(results)
print(results)
