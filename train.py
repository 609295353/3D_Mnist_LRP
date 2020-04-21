from model import CNN
import tensorflow as tf
import data_test as dt
from utils import DataGenerator
import matplotlib.pyplot as plt

logdir = './logs/'
chkpt = './logs/model.ckpt'
# 训练epoch数量
n_epochs = 100
# 每个batch大小
batch_size = 86
# 数据集三维大小
image_x = 16
image_y = 16
image_z = 16
image_c = 1
# 数据集类别数量
n_label = 10
train_acc_list=[]
test_acc_list=[]

class Trainer:

	def __init__(self):
		# 导入数据

		self.x_train, self.y_train, self.x_validation, self.y_validation= dt.getdata()

		with tf.variable_scope('CNN'):
			self.model = CNN()

			self.X = tf.placeholder(tf.float32, [None, image_x,image_y,image_z,image_c], name='X')
			self.y = tf.placeholder(tf.float32, [None, n_label], name='y')
			self.is_training = tf.placeholder(tf.bool, [])
			self.keep_prob1 = tf.placeholder(tf.float32, [])
			self.keep_prob2 = tf.placeholder(tf.float32, [])

			self.activations, self.logits = self.model(self.X,image_x,image_y,image_z,image_c,n_label,self.is_training,self.keep_prob1,self.keep_prob2)
			tf.add_to_collection('is_training', self.is_training)
			tf.add_to_collection('is_training', self.keep_prob1)
			tf.add_to_collection('is_training', self.keep_prob2)
			tf.add_to_collection('LayerwiseRelevancePropagation', self.X)
			for act in self.activations:
				tf.add_to_collection('LayerwiseRelevancePropagation', act)

			self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.y))
			self.optimizer = tf.train.AdamOptimizer().minimize(self.cost, var_list=self.model.params)
			self.out = tf.argmax(self.logits, axis=1)
			tf.add_to_collection("out",self.out)
			self.preds = tf.equal(tf.argmax(self.logits, axis=1), tf.argmax(self.y, axis=1))
			self.accuracy = tf.reduce_mean(tf.cast(self.preds, tf.float32))

		self.cost_summary = tf.summary.scalar(name='Cost', tensor=self.cost)
		self.accuracy_summary = tf.summary.scalar(name='Accuracy', tensor=self.accuracy)

		self.summary = tf.summary.merge_all()

	def run(self):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			self.saver = tf.train.Saver()
			self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

			self.train_batch = DataGenerator(self.x_train, self.y_train, batch_size)
			self.validation_batch = DataGenerator(self.x_validation, self.y_validation, batch_size)
			flag = True

			for epoch in range(n_epochs):
				self.train(sess, epoch)
				self.validate(sess)
				self.saver.save(sess, chkpt)

			self.draw()


    # 训练
	def train(self, sess, epoch):
		n_batches = self.x_train.shape[0] // batch_size
		if self.x_train.shape[0] % batch_size != 0:
			n_batches += 1

		avg_cost = 0
		avg_accuracy = 0
		for batch in range(n_batches):
			x_batch, y_batch = next(self.train_batch)
			_, batch_cost, batch_accuracy, summ = sess.run([self.optimizer, self.cost, self.accuracy, self.summary],
																											feed_dict={self.X: x_batch, self.y: y_batch,self.is_training:True,self.keep_prob1:0.75,self.keep_prob2:0.5})
			avg_cost += batch_cost
			avg_accuracy += batch_accuracy
			self.file_writer.add_summary(summ, epoch * n_batches + batch)

			completion = batch / n_batches
			print_str = '|' + int(completion * 20) * '#' + (19 - int(completion * 20))  * ' ' + '|'
			print('\rEpoch {0:>3} {1} {2:3.0f}% Cost {3:6.4f} Accuracy {4:6.4f}'.format('#' + str(epoch + 1),
				print_str, completion * 100, avg_cost / (batch + 1), avg_accuracy / (batch + 1)), end='')
		print(end=' ')
		train_acc_list.append(avg_accuracy / n_batches)

    # 测试
	def validate(self, sess):
		n_batches = self.x_validation.shape[0] // batch_size
		if self.x_validation.shape[0] % batch_size != 0:
			n_batches += 1

		avg_accuracy = 0
		for batch in range(n_batches):
			x_batch, y_batch = next(self.validation_batch)
			avg_accuracy += sess.run([self.accuracy, ], feed_dict={self.X: x_batch, self.y: y_batch,self.is_training:False,self.keep_prob1:1,self.keep_prob2:1})[0]

		avg_accuracy /= n_batches
		print('Validation Accuracy {0:6.4f}'.format(avg_accuracy))
		test_acc_list.append(avg_accuracy)

	# 训练集测试集精度记录图
	def draw(self):
		plt.plot(range(1,n_epochs+1),train_acc_list,label="train",color="r")
		plt.plot(range(1,n_epochs+1),test_acc_list,label="test",color="b")
		plt.xlabel('epoch')
		plt.ylabel('ACC')
		plt.title('train,test Accuracy')
		plt.legend()
		plt.savefig("./train_test_result.png")
		plt.show()
if __name__ == '__main__':
	Trainer().run()
