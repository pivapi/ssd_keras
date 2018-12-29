from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

# %matplotlib inline

#############################################
# Set a few configuration parameters.
img_height = 300
img_width = 300
n_classes = 20
model_mode = 'inference'
#############################################
# # 1: Build the Keras model
#
# K.clear_session() # Clear previous models from memory.
#
# model = ssd_300(image_size=(img_height, img_width, 3),
#                 n_classes=n_classes,
#                 mode=model_mode,
#                 l2_regularization=0.0005,
#                 scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
#                 aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                                          [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
#                                          [1.0, 2.0, 0.5],
#                                          [1.0, 2.0, 0.5]],
#                 two_boxes_for_ar1=True,
#                 steps=[8, 16, 32, 64, 100, 300],
#                 offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
#                 clip_boxes=False,
#                 variances=[0.1, 0.1, 0.2, 0.2],
#                 normalize_coords=True,
#                 subtract_mean=[123, 117, 104],
#                 swap_channels=[2, 1, 0],
#                 confidence_thresh=0.01,
#                 iou_threshold=0.45,
#                 top_k=200,
#                 nms_max_output_size=400)
#
# # 2: Load the trained weights into the model.
#
# # TODO: Set the path of the trained weights.
# weights_path = 'weights/VGG_VOC0712_SSD_300x300_ft_iter_120000.h5'
#
# model.load_weights(weights_path, by_name=True)
#
# # 3: Compile the model so that Keras won't complain the next time you load it.
#
# adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
#
# model.compile(optimizer=adam, loss=ssd_loss.compute_loss)
# # TODO: Set the path to the `.h5` file of the model to be loaded.
# model_path = 'ssd300_pascal_07+12/ssd300_pascal_07+12_epoch-110_loss-3.5820_val_loss-3.0160.h5'
#
# # We need to create an SSDLoss object in order to pass that to the model loader.
# ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)
#
# K.clear_session() # Clear previous models from memory.
#
# model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
#                                                'L2Normalization': L2Normalization,
#                                                'DecodeDetections': DecodeDetections,
#                                                'compute_loss': ssd_loss.compute_loss})

#############################################
dataset = DataGenerator()

# TODO: Set the paths to the dataset here.
Pascal_VOC_dataset_images_dir = '/home/piva/Datasets/PASCAL-VOC/VOCdevkit/VOC2007_Test/JPEGImages/'
Pascal_VOC_dataset_annotations_dir = '/home/piva/Datasets/PASCAL-VOC/VOCdevkit/VOC2007_Test/Annotations/'
Pascal_VOC_dataset_image_set_filename = '/home/piva/Datasets/PASCAL-VOC/VOCdevkit/VOC2007_Test/ImageSets/Main/test.txt'

# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

dataset.parse_xml(images_dirs=[Pascal_VOC_dataset_images_dir],
                  image_set_filenames=[Pascal_VOC_dataset_image_set_filename],
                  annotations_dirs=[Pascal_VOC_dataset_annotations_dir],
                  classes=classes,
                  include_classes='all',
                  exclude_truncated=False,
                  exclude_difficult=False,
                  ret=False)

#############################################
evaluator = Evaluator(model=model,
                      n_classes=n_classes,
                      data_generator=dataset,
                      model_mode=model_mode)

results = evaluator(img_height=img_height,
                    img_width=img_width,
                    batch_size=8,
                    data_generator_mode='resize',
                    round_confidences=False,
                    matching_iou_threshold=0.5,
                    border_pixels='include',
                    sorting_algorithm='quicksort',
                    average_precision_mode='sample',
                    num_recall_points=11,
                    ignore_neutral_boxes=True,
                    return_precisions=True,
                    return_recalls=True,
                    return_average_precisions=True,
                    verbose=True)

mean_average_precision, average_precisions, precisions, recalls = results
#############################################
#############################################

#############################################
#############################################
#############################################



