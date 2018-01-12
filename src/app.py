import argparse
import sys
from flask import Flask
import tensorflow as tf
from model import Model

tf.app.flags.DEFINE_boolean('train', False, '--train launch the app in training mode, ignore for serving')
tf.app.flags.DEFINE_integer('iteration', 2000, '')
FLAGS = tf.app.flags.FLAGS  


def main(_):
  model = Model()
  print FLAGS
  if FLAGS.train:
    model.train(itr=int(sys.argv[3]))
    model.save_model(tosave = True, topath=sys.argv[2])
    sys.exit(0)

  
if __name__ == '__main__':
  print "Number of arguments: ", len(sys.argv)
  print "The arguments are: " , str(sys.argv)
  print "train or not ", sys.argv[1]
  print "volume path to save", sys.argv[2]
  print "iterations", sys.argv[3]
  tf.app.run()

