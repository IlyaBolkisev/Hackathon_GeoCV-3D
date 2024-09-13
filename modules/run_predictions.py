import os

import cv2
import numpy as np

import onnxruntime as ort


def warmup_models(path_to_folder):
    models = {}
    for fn in os.listdir(path_to_folder):
        name = fn.split('.')[0]
        models[name] = ort.InferenceSession(f"{path_to_folder}/{fn}", providers=['CUDAExecutionProvider'])
        input_dict = {inp.name: np.random.rand(1, *inp.shape[1:]).astype('float32')
                      for inp in models[name].get_inputs()}
        output_names = [output.name for output in models[name].get_outputs()]
        models[name].run(output_names, input_dict)
    return models


def preprocess_img(img, shape_to=(137, 137)):
    img_resized = cv2.resize(img, (shape_to[0], shape_to[1]), interpolation=cv2.INTER_CUBIC)

    img_array = img_resized / (img_resized.max() + 1e-10)
    img_tensor = np.expand_dims(np.transpose(img_array, (2, 0, 1)), axis=0)
    return img_tensor.astype('float32')


def run_predictions(left, right, models):
    left, right = preprocess_img(left), preprocess_img(right)

    input_dict = {'left': left, 'right': right}
    output_names = [output.name for output in models['disp'].get_outputs()]
    disp_l, disp_r = models['disp'].run(output_names, input_dict)

    left_ = np.concatenate((left, disp_l), axis=1)
    right_ = np.concatenate((right, disp_r), axis=1)

    input_dict = {'rgbd': left_}
    output_names = [output.name for output in models['enc'].get_outputs()]
    left_, left_corr = models['enc'].run(output_names, input_dict)
    input_dict = {'rgbd': right_}
    right_, right_corr = models['enc'].run(output_names, input_dict)

    input_dict = {'left': left_corr, 'right': right_corr}
    output_names = [output.name for output in models['cor'].get_outputs()]
    corr_ = models['cor'].run(output_names, input_dict)[0]

    input_dict = {'left': left_.reshape((1, 128, 4, 4, 4)), 'right': right_.reshape((1, 128, 4, 4, 4)),
                  'cor': corr_.reshape((1, 64, 4, 4, 4))}
    output_names = [output.name for output in models['dec'].get_outputs()]
    volume = models['dec'].run(output_names, input_dict)[0]

    return disp_l[0], disp_r[0], volume[0]

