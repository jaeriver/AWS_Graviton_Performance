import json
import boto3
import numpy as np
from PIL import Image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import time
import argparse

bucket_name = 'imagenet-sample'
bucket_ensemble = 'lambda-ensemble'

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--batch_list',
                      nargs='+',
                      required=True)
batch_list = parser.parse_args().batch_list

model_name = 'resnet50'
model_path = 'model/' + model_name
model = load_model(model_path, compile=True)


s3 = boto3.resource('s3')
s3_client = boto3.client('s3')

filenames = [file.key for file in bucket.objects.all() if len(file.key.split('/')[1]) > 1]

def read_image_from_s3(filename):
    bucket = s3.Bucket(bucket_name)
    object = bucket.Object(filename)
    response = object.get()
    file_stream = response['Body']
    img = Image.open(file_stream)
    img.convert('RGB')
    return img


def filenames_to_input(file_list):
    imgs = []
    for file in file_list:
        img = read_image_from_s3(file)
        img = img.resize((224, 224), Image.ANTIALIAS)
        img = np.array(img)
        # if image is grayscale, convert to 3 channels
        if len(img.shape) != 3:
            img = np.repeat(img[..., np.newaxis], 3, -1)
        # batchsize, 224, 224, 3
        img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
        img = preprocess_input(img)
        imgs.append(img)

    batch_imgs = np.vstack(imgs)
    return batch_imgs

def decode_predictions(preds, top=1):
    # get imagenet_class_index.json from container directory
    with open('imagenet_class_index.json') as f:
        CLASS_INDEX = json.load(f)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results

def inference_model(batch_imgs):
    pred_start = time.time()
    result = model.predict(batch_imgs)
    pred_time = time.time() - pred_start

    result = decode_predictions(result)
    results = []
    for single_result in result:
        single_result = [(img_class, label, str(round(acc * 100, 4)) + '%') for img_class, label, acc in single_result]
        results.append(single_result)
    return results, pred_time


def perform(batch_size):
    file_list = filenames[:batch_size]
    batch_imgs = filenames_to_input(file_list)

    total_start = time.time()
    result, pred_time = inference_model(batch_imgs)
    total_time = time.time() - total_start
    print(results)
    return {
        'model_name': model_name,
        'batch_size': batch_size,
        'total_time': total_time,
        'pred_time': pred_time,
    }

if __name__ == "__main__":
    for batch_size in batch_list:
        print(perform(batch_size))
