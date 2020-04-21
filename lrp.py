import data_test as dt
from tensorflow.python.ops    import gen_nn_ops
import  plot  as plot

import numpy                as np
import tensorflow         as tf
import matplotlib.pyplot    as plt

logdir = './logs/'
chkpt = './logs/model.ckpt'
resultsdir = './results/'

class LayerwiseRelevancePropagation:

  def __init__(self):
    self.epsilon = 1e-10


    with tf.Session() as sess:
      saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
      saver.restore(sess, tf.train.latest_checkpoint(logdir))

      weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='CNN')
      self.activations = tf.get_collection('LayerwiseRelevancePropagation')
    self.is_training = tf.get_collection('is_training')[0]
    self.keep_prob1 = tf.get_collection('is_training')[1]
    self.keep_prob2 = tf.get_collection('is_training')[2]
    self.out = tf.get_collection("out")[0]
    self.X = self.activations[0]

    self.act_weights = {}
    for act in self.activations[2:]:
      for wt in weights:
        name = act.name.split('/')[2]
        if name == wt.name.split('/')[2]:
          if name not in self.act_weights:
            self.act_weights[name] = wt

    self.activations = self.activations[:0:-1]
    self.relevances = self.get_relevances()

  # 获取各层的relevance
  def get_relevances(self):
    relevances = [self.activations[0], ]

    for i in range(1, len(self.activations)):
      name = self.activations[i - 1].name.split('/')[2]
      if 'output' in name or 'fc' in name:
        relevances.append(self.backprop_fc(name, self.activations[i], relevances[-1]))
      elif 'flatten' in name:
        relevances.append(self.backprop_flatten(self.activations[i], relevances[-1]))
      elif 'max_pool' in name:
        s = name.split('_')
        relevances.append(self.backprop_max_pool3d(self.activations[i], relevances[-1],s[2]))
      elif 'conv' in name:
        s = name.split('_')
        strides=[1,1,1,1,1]
        print(name)
        if(s[2]=='2'):
          strides = [1, 2, 2, 2, 1]
        relevances.append(self.backprop_conv3d(name, self.activations[i], relevances[-1],tf.shape(self.activations[i]),s[1],strides))
      else:
        raise 'Error parsing layer!'

    return relevances

  # 全连接层的lrp
  def backprop_fc(self, name, activation, relevance):
    w = self.act_weights[name]
    w_pos = tf.maximum(0.0, w)
    z = tf.matmul(activation, w_pos) + self.epsilon
    s = relevance / z
    c = tf.matmul(s, tf.transpose(w_pos))
    return c * activation

  # flatten层的lrp
  def backprop_flatten(self, activation, relevance):
    shape = activation.get_shape().as_list()
    shape[0] = -1
    return tf.reshape(relevance, shape)

  # maxpooling层的lrp
  def backprop_max_pool3d(self, activation, relevance,padding, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1]):
    z = tf.nn.max_pool3d(activation, ksize, strides, padding=padding) + self.epsilon
    s = relevance / z
    c = gen_nn_ops.max_pool3d_grad(activation, z, s, ksize, strides, padding=padding)
    return c * activation

  # 3d卷积层的lrp
  def backprop_conv3d(self, name, activation, relevance,out_shape,padding, strides=[1, 1, 1, 1, 1]):
    w = self.act_weights[name]
    w_pos = tf.maximum(0.0, w)
    z = tf.nn.conv3d(activation, w_pos, strides, padding=padding) + self.epsilon
    print("activation:",activation.shape)
    print("relevance:",relevance.shape)
    print("z",z.shape)
    print("outshape",out_shape)
    s = relevance / z
    print("w:",w_pos.shape)
    print("s:",s.shape)
    c = tf.nn.conv3d_transpose(s, w_pos, out_shape, strides, padding=padding)
    print("c:",c.shape)
    return c*activation

  # 获取heatmap
  def get_heatmap(self,image_x,image_y,image_z):
    samples,lbs = dt.getsamples()
    with tf.Session() as sess:
      saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
      saver.restore(sess, tf.train.latest_checkpoint(logdir))
      heatmaps = []
      for i in range(samples.shape[0]):
        heatmap= sess.run(self.relevances[-1], feed_dict={self.X: samples[i:i+1],self.is_training:False,self.keep_prob1:1,self.keep_prob2:1})[0].reshape(image_x, image_y,image_z)
        pred = sess.run(self.out, feed_dict={self.X: samples[i:i+1],self.is_training:False,self.keep_prob1:1,self.keep_prob2:1})
        # 将heatmap中像素值分布到[0,1]的区间
        heatmap = heatmap / (np.max(heatmap)) * 255
        # 切片显示
        # plot.show_slices(samples[i].reshape([16,16,16]),heatmap,pred,lbs[i])
        # 切片显示
        plot.show_heatmap(heatmap,pred,lbs[i])
        plot.plot3d(heatmap,pred,lbs[i])
        heatmaps.append(heatmap)
    return heatmaps

  def test(self):
    samples = dt.getsamples()

    print("hehehe")
    with tf.Session() as sess:
      saver = tf.train.import_meta_graph('{0}.meta'.format(chkpt))
      saver.restore(sess, tf.train.latest_checkpoint(logdir))
      R, ac= sess.run([self.relevances,self.activations], feed_dict={self.X: samples[0],self.is_training:False,self.keep_prob1:1,self.keep_prob2:1})
      for a in ac:
        print(a.shape)
        print(np.max(a))
      i=0
      for r in R:
        name = self.activations[i].name.split('/')[2]
        print(name)
        print(r.sum())
        i+=1

if __name__ == '__main__':
  lrp = LayerwiseRelevancePropagation()
  # lrp.test()
  image_x = 16
  image_y = 16
  image_z = 16
  heatmaps = np.array(lrp.get_heatmap(image_x,image_y,image_z))
  # for i in range(heatmaps.shape[0]):
  #   fig = plt.figure()
  #   ax = fig.add_subplot(111)
  #   ax.axis('off')
  #   # heatmaps[i] *=255*256*4
  #   h=(heatmaps[i][:,:,0]+heatmaps[i][:,:,1]+heatmaps[i][:,:,2])/3*256*256*255
  #   print(np.max(heatmaps[i]),np.min(heatmaps[i]))
  #   ax.imshow(h, cmap='Reds', interpolation='bilinear')
  #   plt.show()
  #   fig.savefig('{0}{1}.jpg'.format(resultsdir,i))