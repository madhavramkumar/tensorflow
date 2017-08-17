# import TensorFlow
import tensorflow as tf

sess = tf.Session()

# Verify we can print a string
hello = tf.constant('Hello, Tensorflow!')
print(sess.run(hello))

#Perform some simple math
a = tf.constant(20)
b = tf.constant(10)
print('a + b = {0}'.format(sess.run(a + b)))

