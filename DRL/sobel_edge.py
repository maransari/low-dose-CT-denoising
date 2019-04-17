from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 13:15:54 2018

@author: mansari
"""
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of image ops."""



import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.util.tf_export import tf_export


def sobel(image):
  """Returns a tensor holding Sobel edge maps.

  Arguments:
    image: Image tensor with shape [batch_size, h, w, d] and type float32 or
    float64.  The image(s) must be 2x2 or larger.

  Returns:
    Tensor holding edge maps for each channel. Returns a tensor with shape
    [batch_size, h, w, d, 2] where the last two dimensions hold [[dy[0], dx[0]],
    [dy[1], dx[1]], ..., [dy[d-1], dx[d-1]]] calculated using the Sobel filter.
  """
  # Define vertical and horizontal Sobel filters.
  static_image_shape = image.get_shape()
  image_shape = array_ops.shape(image)
  kernels = [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[0, 1, 2], [-1, 0, 1], [-2, -1, 0]],
             [[-2, -1, 0], [-1, 0, 1], [0, 1, 2]]]
  num_kernels = len(kernels)
  kernels = np.transpose(np.asarray(kernels), (1, 2, 0))
  kernels = np.expand_dims(kernels, -2)
  kernels_tf = constant_op.constant(kernels, dtype=image.dtype)

  kernels_tf = array_ops.tile(kernels_tf, [1, 1, image_shape[-1], 1],
                              name='sobel_filters')

  # Use depth-wise convolution to calculate edge maps per channel.
  pad_sizes = [[0, 0], [1, 1], [1, 1], [0, 0]]
  padded = array_ops.pad(image, pad_sizes, mode='REFLECT')

  # Output tensor has shape [batch_size, h, w, d * num_kernels].
  strides = [1, 1, 1, 1]
  output = nn.depthwise_conv2d(padded, kernels_tf, strides, 'VALID')

  # Reshape to [batch_size, h, w, d, num_kernels].
  shape = array_ops.concat([image_shape, [num_kernels]], 0)
  output = array_ops.reshape(output, shape=shape)
  output.set_shape(static_image_shape.concatenate([num_kernels]))
  return output