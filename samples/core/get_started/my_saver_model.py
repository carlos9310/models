import tensorflow as tf


class DNNConfig(object):
    num_columns = 4         # 特征列数
    num_classes = 3         # 类别数
    hidden_units = 10       # 隐层单元数
    hidden_layers = 2       # 隐层数
    learning_rate = 0.1     # 学习率
    batch_size = 100        # 批次大小
    train_steps = 1000      # 训练步数


class DNN(object):
    def __init__(self):
        self.input_features = tf.placeholder(tf.float32, [None, DNNConfig.num_columns], name='input_features')
        self.input_labels = tf.placeholder(tf.int64, [None], name='input_labels')
        self.dnn() 

    def dnn(self):
        with tf.name_scope("hidden"):
            hidden_1 = tf.layers.dense(self.input_features, units=DNNConfig.hidden_units, activation=tf.nn.relu)
            hidden_2 = tf.layers.dense(hidden_1, units=DNNConfig.hidden_units, activation=tf.nn.relu)

        with tf.name_scope("output"):
            self.logits = tf.layers.dense(hidden_2, DNNConfig.num_classes, activation=None,trainable=True)
            with tf.name_scope("probability"):
                self.logits_prob = tf.nn.softmax(self.logits)
            with tf.name_scope("predict_result"):
                self.predicted_classes = tf.argmax(self.logits, 1)
                self.class_ids = self.predicted_classes[:, tf.newaxis]

        with tf.name_scope("optimize"):
            self.cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels=self.input_labels, logits=self.logits)
            self.train_op = tf.train.AdagradOptimizer(learning_rate=DNNConfig.learning_rate).minimize(self.cross_entropy_loss) # , global_step=tf.train.get_global_step()

        with tf.name_scope("accuracy"):
            accuracy = tf.metrics.accuracy(labels=self.input_labels,
                                           predictions=self.predicted_classes,
                                           name='acc_op')
            self.acc = accuracy[1]
            # 准确率，与上面是等价的
            # correct_pred = tf.equal(self.predicted_classes, self.input_labels)
            # self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
