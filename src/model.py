#import azure_blob_helper
import os 
import numpy as np
import tensorflow as tf
import tensorflow.contrib.lookup
from tensorflow.examples.tutorials.mnist import input_data
import cPickle as pickle
import urllib
import json
import numpy as np
from waitress import serve
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
from google.protobuf.json_format import MessageToJson
import shutil

save_dir="/tmp/ckp/"

class Model:
  #x = tf.placeholder(tf.float32, [None, 784])
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')

  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b
  sess = tf.InteractiveSession()



  def train(self,itr = 1000):
    mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=self.y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    tf.global_variables_initializer().run()

    print(itr)
    self.values, self.indices = tf.nn.top_k(self.y, 10)
    table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in xrange(10)]))
    self.prediction_classes = table.lookup(tf.to_int64(self.indices))
    for _ in range(itr):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      self.sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print "=================================="
    print 'training accuracy %g' % self.sess.run(accuracy, feed_dict={self.x: mnist.test.images, y_: mnist.test.labels})
    print "=================================="
    print(self.sess.run(accuracy, feed_dict={self.x: mnist.test.images, y_: mnist.test.labels}))

  def predict(self, x):
    feed_dict = {self.x: x}
    prediction = self.sess.run(tf.nn.softmax(self.y), feed_dict)
    return prediction

  def save_model(self, tosave = True, topath=save_dir):  
    if os.path.isdir(topath) == True:
      shutil.rmtree(topath)
    builder = tf.saved_model.builder.SavedModelBuilder(topath)

    classification_inputs = tf.saved_model.utils.build_tensor_info(self.serialized_tf_example)
    classification_outputs_classes = tf.saved_model.utils.build_tensor_info(self.prediction_classes)
    classification_outputs_scores = tf.saved_model.utils.build_tensor_info(self.values)

    classification_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={ tf.saved_model.signature_constants.CLASSIFY_INPUTS:classification_inputs },
              outputs={ tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES: classification_outputs_classes,
                  tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES: classification_outputs_scores
              },
              method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    tensor_info_x = tf.saved_model.utils.build_tensor_info(self.x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(self.y)

    prediction_signature = (
          tf.saved_model.signature_def_utils.build_signature_def(
              inputs={'images': tensor_info_x},
              outputs={'scores': tensor_info_y},
              method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
          self.sess, [tf.saved_model.tag_constants.SERVING],
          signature_def_map={ 'predict_images': prediction_signature,
              tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: classification_signature,
          },
          legacy_init_op=legacy_init_op)

    builder.save()



