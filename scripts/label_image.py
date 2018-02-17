# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import time

import cv2
import numpy as np
import tensorflow as tf

asegmentsHeight = 8
asegmentsWidth = 8
def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


'''
def split_image(im_name):
    print "Reading Image"
    image = cv2.imread(im_name)
    #cv2.imshow('image',image)
    #cv2.waitKey(0) throws error at runtime
    
    height,width = image.shape[:2]
    print "Image size: "
    print image.shape[:2]
    heightSeg = height/asegmentsHeight
    widthSeg  = width/asegmentsWidth
    
    i = 1
    for x in xrange(1, asegmentsWidth+1):
        for y in xrange(1,asegmentsHeight+1):
            #print str(heightSeg*(y-1)) + " " + str(heightSeg*y)
            #print str(widthSeg*(x-1)) + " " + str(widthSeg*x)
            seg_image = image[heightSeg*(y-1):heightSeg*y,widthSeg*(x-1):widthSeg*x]    
            #show and write image segments               
            name = 'segment' + str(i)
            #cv2.imshow(name,segImage)
            #cv2.waitKey(1)
            
            #write file
            cv2.imwrite( str(name) + '.jpeg',seg_image)

            #label image with classifier
            print "Labels for %s: " % name
            #label(name,seg_image)

            i=i+1

    cv2.destroyAllWindows()
    return name
'''


def read_tensor_from_image_file(file_name, input_height=299, input_width=299,
				input_mean=0, input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, channels = 3,
                                       name='png_reader')
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.image.decode_gif(file_reader,
                                                  name='gif_reader'))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name='bmp_reader')
  else:
    image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0);
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

if __name__ == "__main__":
  file_name = "tf_files/flower_photos/daisy/3475870145_685a19116d.jpg"
  model_file = "tf_files/retrained_graph.pb"
  label_file = "tf_files/retrained_labels.txt"
  input_height = 224
  input_width = 224
  input_mean = 128
  input_std = 128
  input_layer = "input"
  output_layer = "final_result"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer

  graph = load_graph(model_file)
  
  print("Reading Image")
  image = cv2.imread(file_name)
  #cv2.imshow('image',image)
  #cv2.waitKey(0) throws error at runtime

  height,width = image.shape[:2]
  print("Image size: ")
  print(image.shape[:2])
  heightSeg = height/asegmentsHeight
  widthSeg  = width/asegmentsWidth
  image_res = image.copy()
  i = 0
  for x1 in xrange(1, asegmentsWidth+1):
      for y1 in xrange(1,asegmentsHeight+1):
          print( str(heightSeg*(y1-1)) + " " + str(heightSeg*y1))
          print (str(widthSeg*(x1-1)) + " " + str(widthSeg*x1))
          seg_image = image[int(heightSeg*(y1-1)):int(heightSeg*y1),int(widthSeg*(x1-1)):int(widthSeg*x1)]    
          #show and write image segments               
          f_name = 'segment' + str(i) + '.jpeg'
          #cv2.imshow(name,segImage)
          #cv2.waitKey(1)
          
          #write file
          cv2.imwrite( f_name ,seg_image)

          #label image with classifier
          print("Labels for %s: " % f_name)
          #label(name,seg_image)
          i=i+1
          print("i is:",i)
          t = read_tensor_from_image_file(f_name,
                                          input_height=input_height,
                                          input_width=input_width,
                                          input_mean=input_mean,
                                          input_std=input_std)

          input_name = "import/" + input_layer
          output_name = "import/" + output_layer
          input_operation = graph.get_operation_by_name(input_name);
          output_operation = graph.get_operation_by_name(output_name);

          with tf.Session(graph=graph) as sess:
            start = time.time()
            results = sess.run(output_operation.outputs[0],
                              {input_operation.outputs[0]: t})
            end=time.time()
          results = np.squeeze(results)

          top_k = results.argsort()[-5:][::-1]
          labels = load_labels(label_file)

          print('\nEvaluation time (1-image): {:.3f}s\n'.format(end-start))          
          
          for labl in top_k:
            print(labels[labl], results[labl])
            
          high = np.argmax(results)  
          new_name = 'results1/segment' + str(i) + str(labels[high]) + '.jpeg'
          cv2.imwrite( new_name ,seg_image)
          print("image res:",image_res[0,0])
          
          for aheight in xrange(int(heightSeg*(y1-1)),int(heightSeg*y1)):
            for awidth in xrange(int(widthSeg*(x1-1)),int(widthSeg*x1)):
              if labels[high] =='empty':
                continue#image_res[aheight,awidth] = [0,0,0]
              elif(labels[high]=='full'):
                image_res[aheight,awidth] = [0,0,0]#[255,255,255]
              else:
                print("wrong label")
                
  cv2.imwrite("out.jpeg",image_res)          
  
  #cv2.destroyAllWindows()  
