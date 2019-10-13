import tensorflow as tf
import os
import numpy as np
import argparse
import iris_data
from my_saver_model import DNN, DNNConfig

save_dir = 'my_saver'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_path = os.path.join(save_dir, 'saver.ckpt')  # 保存为checkpoint格式

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default=None, type=str,
                    help='train or eval mode')


def train(train_x, train_y, batch_size):
    model = DNN()

    train_dataset = iris_data.train_input_fn(train_x, train_y, batch_size)
    iterator = train_dataset.make_one_shot_iterator()

    train_next_example, train_next_label = iterator.get_next()

    train_input = tf.feature_column.input_layer(train_next_example, my_feature_columns)

    # 配置 Saver
    saver = tf.train.Saver()

    tvars = tf.trainable_variables()
    for var in tvars:
        print(f'name = {var.name}, shape = {var.shape}')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 获取图中所有节点(node)名称(数据节点/计算节点)，按名称取tensor时需指定索引号，无索引号为op。具体示例如下：
        # The name 'input_features' refers to an Operation, not a Tensor. Tensor names must be of the form "<op_name>:<output_index>".
        print([n.name for n in tf.get_default_graph().as_graph_def().node])
        for train_steps in range(DNNConfig.train_steps):
            train_features, train_labels = sess.run([train_input, train_next_label])
            _, acc_train, loss,input_features = sess.run(
                [model.train_op, model.acc, model.cross_entropy_loss,model.input_features],
                feed_dict={model.input_features: train_features, model.input_labels: train_labels})

            print(f'train_steps:{train_steps},acc_train:{acc_train}') # ,labels:{labels}
            # for e in tvars:
            #     print("name: ", e.name)
            #     # print("value: ", sess.run(sess.graph.get_tensor_by_name(e.name)))
            #     print("value: ", sess.run(e.value()))
        print(f'train_steps = {train_steps} train end and start save model')

        # save
        saver.save(sess=sess, save_path=model_path)
        print(f'end save model,the model is saved in {model_path}')


def eval(test_x, test_y, batch_size, model_path):
    print(f'loading model from {model_path}')
    eval_dataset = iris_data.eval_input_fn(test_x, test_y, batch_size)
    eval_iterator = eval_dataset.make_one_shot_iterator()

    eval_next_example, eval_next_label = eval_iterator.get_next()

    eval_input = tf.feature_column.input_layer(eval_next_example, my_feature_columns)

    with tf.Session() as sess:
        # import meta graph and restore variables
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)
        # get graph
        graph = tf.get_default_graph()

        # in this case,we need to know the tensor name define in the graph during training
        # and using get_tensor_by_name here
        input_features_tensor = graph.get_tensor_by_name('input_features:0')
        predicted_classes_tensor = graph.get_tensor_by_name('output/predict_result/predicted_classes:0')

        # 验证正常加载了模型参数
        dense_2_bias = sess.run('dense_2/bias:0')
        dense_1_bias = sess.run('dense_1/bias:0')

        print(f'dense_2/bias:0 =  {dense_2_bias}')
        print(f'dense_1/bias:0 =  {dense_1_bias}')

        eval_features = sess.run(eval_input)

        pre_id = sess.run(predicted_classes_tensor, feed_dict={input_features_tensor: eval_features})

        # pre_id = np.array(pre_id).reshape(-1)
        print(f'pre_id:{pre_id}')
        # # print(pre_logits)
        # # print(pre_probabilities)
        # real_id = np.array(test_y).reshape(-1,1)
        real_id = np.array(test_y)
        print(f'rea_id:{real_id}')

        # 整个评估数据集的准确率
        correct_pred = tf.equal(real_id, pre_id)
        eval_acc = sess.run(tf.reduce_mean(tf.cast(correct_pred, tf.float32)))
        print(f'eval set accuracy:{eval_acc}')


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.mode or args.mode != 'train' or args.mode != 'eval':
        print('you must set the mode like this: --mode <train / eval>')

    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    if args.mode == 'train':
        train(train_x, train_y, DNNConfig.batch_size)

    if args.mode == 'eval':
        eval(test_x, test_y, DNNConfig.batch_size, model_path)
