import tensorflow as tf
from keras import regularizers

class CNN:
  
  def __init__(self, name='CNN'):
    self.name = name


  def convlayer(self, input, shape,is_training=False,name='conv',strides=[1, 1, 1, 1, 1],isfirst=False):
    regL = 0.01
    if isfirst:
      conv = tf.nn.relu(tf.layers.conv3d(inputs=input, filters=shape[-1], kernel_size=[3, 3, 3], padding='same',
                                         kernel_regularizer=regularizers.l2(regL), name=name))
    else:
      conv = tf.nn.relu(tf.layers.conv3d(inputs=input, filters=shape[-1], kernel_size=[3, 3, 3], padding='same',
                                         kernel_regularizer=regularizers.l2(regL),
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(), name=name))

    return  conv
  
  def fclayer(self, input, shape,is_training=False, name="fc", prop=True,isoutput=False):
    regL = 0.01
    if isoutput:
      fc = tf.layers.dense(inputs=input, units=shape[-1], name=name)
    else:
      fc = tf.nn.relu(tf.layers.dense(inputs=input, units=shape[-1], kernel_regularizer=regularizers.l2(regL),
                                      bias_regularizer=regularizers.l2(regL), name=name))
    return fc

  def __call__(self, images,image_x,image_y,image_z,image_c,n_label,is_training,keep_prob1,keep_prob2, reuse=False):
    with tf.variable_scope(self.name):
      
      if reuse:
        scope.reuse_variables()
      
      activations = []

      with tf.variable_scope('input'):
        images = tf.reshape(images, [-1, image_x, image_y,image_z, image_c], name='input')
        activations += [images, ]

      with tf.variable_scope('conv1_SAME_1'):
        conv1 = self.convlayer(images, [3, 3, 3, image_c, 32],is_training, 'conv1_SAME_1',isfirst=True)
        activations += [conv1, ]

      with tf.variable_scope('conv2_SAME_1'):
        conv2 = self.convlayer(conv1, [3, 3, 3, 32, 64], is_training, 'conv2_SAME_1')
        activations += [conv2, ]

      with tf.variable_scope('max_pool1_VALID'):
        max_pool1 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2, 2, 2], strides=2, name='max_pool1_VALID')
        activations += [max_pool1, ]

      with tf.variable_scope('conv3_SAME_1'):
        conv3 = self.convlayer(max_pool1, [3, 3, 3, 64, 128], is_training, 'conv3_SAME_1')
        activations += [conv3, ]

      with tf.variable_scope('conv4_SAME_1'):
        conv4 = self.convlayer(conv3, [3, 3, 3, 128, 256], is_training, 'conv4_SAME_1')
        activations += [conv4, ]

      with tf.variable_scope('max_pool2_VALID'):
        max_pool2 = tf.layers.max_pooling3d(inputs=conv4, pool_size=[2, 2, 2], strides=2, name='max_pool2_VALID')
        activations += [max_pool2, ]

      with tf.variable_scope('flatten'):
        flatten = tf.contrib.layers.flatten(max_pool2)
        activations += [flatten, ]

      with tf.variable_scope('dropout3'):
        dropout3 = tf.layers.dropout(inputs=flatten, rate=1-keep_prob1,training=is_training, name='dropout3')

      with tf.variable_scope('fc1'):
        n_in = int(flatten.get_shape()[1])
        fc1 = self.fclayer(dropout3, [n_in, 4096],is_training, 'fc1')
        activations += [fc1, ]


      with tf.variable_scope('dropout1'):
        dropout1 = tf.layers.dropout(inputs=fc1, rate=1-keep_prob2,training=is_training, name='dropout1')

      with tf.variable_scope('fc2'):
        fc2 = self.fclayer(dropout1, [4096, 1024],is_training, 'fc2')
        activations += [fc2, ]

      with tf.variable_scope('dropout2'):
        dropout2 = tf.layers.dropout(inputs=fc2, rate=1-keep_prob2,training=is_training, name='dropout2')

      with tf.variable_scope('output'):
        logits = self.fclayer(dropout2, [1024, n_label], is_training, 'fc4', isoutput=True)
        preds = tf.nn.softmax(logits, name='output')
        activations += [preds, ]

      return activations, logits
    
  @property
  def params(self):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
    