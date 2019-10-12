import tensorflow as tf
import os
import numpy as np
import time
import argparse
import iris_data
from my_saver_model import DNN, DNNConfig

model_base_path = 'my_savedmodel'

model_path = os.path.join(model_base_path, str(int(time.time())))

tag = tf.saved_model.tag_constants.SERVING

key_my_signature = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY


parser = argparse.ArgumentParser()

parser.add_argument('--mode', default=None, type=str,
                    help='train or eval mode')
parser.add_argument('--model_dir', default=None, type=str,
                    help='dir of exported model for eval')


def train(train_x, train_y, batch_size):
    model = DNN()
    train_dataset = iris_data.train_input_fn(train_x, train_y, batch_size)
    iterator = train_dataset.make_one_shot_iterator()

    train_next_example, train_next_label = iterator.get_next()

    train_input = tf.feature_column.input_layer(train_next_example, my_feature_columns)

    tvars = tf.trainable_variables()
    for var in tvars:
        print(f'name = {var.name}, shape = {var.shape}')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for train_steps in range(DNNConfig.train_steps):
            train_features, train_labels = sess.run([train_input,train_next_label])
            _, acc_train, loss,input_features = sess.run(
                [model.train_op, model.acc, model.cross_entropy_loss,model.input_features],
                feed_dict={model.input_features:train_features,model.input_labels:train_labels}
            )

            # if train_steps % 100 == 0:
            print(f'train_steps:{train_steps},acc_train:{acc_train}') # ,labels:{labels}
            # for e in tvars:
            #     print("name: ", e.name)
            #     # print("value: ", sess.run(sess.graph.get_tensor_by_name(e.name)))
            #     print("value: ", sess.run(e.value()))
        print(f'train_steps = {train_steps} train end and start save model')

        # savedmodel

        # construct saved model builder
        builder = tf.saved_model.builder.SavedModelBuilder(model_path)

        # build inputs and outputs dict,enable us to customize the inputs and outputs tensor name
        # when using the model, we don't need to care the tensor name define in the original graph

        inputs = {'examples': tf.saved_model.utils.build_tensor_info(model.input_features)}

        outputs = {
            'class_ids': tf.saved_model.utils.build_tensor_info(model.class_ids),
            'logits': tf.saved_model.utils.build_tensor_info(model.logits),
            'probabilities': tf.saved_model.utils.build_tensor_info(model.logits_prob)
        }
        method_name = tf.saved_model.signature_constants.PREDICT_METHOD_NAME

        # builder a signature
        my_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs, method_name)

        # add meta graph and variables
        builder.add_meta_graph_and_variables(sess, [tag],
                                             {key_my_signature: my_signature})

        # add_meta_graph method need add_meta_graph_and_variables method been invoked before
        # builder.add_meta_graph(['MODEL_SERVING'], signature_def_map={'my_signature': my_signature})

        # save the model
        # builder.save(as_text=True)
        builder.save()

        print(f'end save model,the model is saved in {model_path}')


def eval(test_x, test_y, batch_size,export_model_path):
    print(f'loading model from {export_model_path}')
    eval_dataset = iris_data.eval_input_fn(test_x, test_y, batch_size)
    eval_iterator = eval_dataset.make_one_shot_iterator()

    eval_next_example, eval_next_label = eval_iterator.get_next()

    eval_input = tf.feature_column.input_layer(eval_next_example, my_feature_columns)

    with tf.Session() as sess:
        # load model
        meta_graph_def = tf.saved_model.loader.load(sess, [tag], export_model_path)

        # get signature
        signature = meta_graph_def.signature_def

        # get tensor name
        in_tensor_name = signature[key_my_signature].inputs['examples'].name
        out_tensor_class_ids = signature[key_my_signature].outputs['class_ids'].name
        out_tensor_logits = signature[key_my_signature].outputs['logits'].name
        out_tensor_probabilities = signature[key_my_signature].outputs['probabilities'].name

        # get tensor
        # in_tensor = sess.graph.get_tensor_by_name(in_tensor_name)
        out_class_ids = sess.graph.get_tensor_by_name(out_tensor_class_ids)
        out_logits = sess.graph.get_tensor_by_name(out_tensor_logits)
        out_probabilities = sess.graph.get_tensor_by_name(out_tensor_probabilities)

        eval_features = sess.run(eval_input)

        eval_features_0 = [eval_features[0]]

        # tvars = tf.trainable_variables()
        # for var in tvars:
        #     print(f'name = {var.name}, value = {sess.run(var.value())}')

        # 验证正常加载了模型参数
        dense_2_bias = sess.run('dense_2/bias:0')
        dense_1_bias = sess.run('dense_1/bias:0')

        # dense_2_bias = sess.graph.get_tensor_by_name('dense_2/bias:0')
        # dense_1_bias = sess.graph.get_tensor_by_name('dense_1/bias:0')
        # dense_2_bias, dense_1_bias = sess.run([dense_2_bias,dense_1_bias])
        print(f'dense_2/bias:0 =  {dense_2_bias}')
        print(f'dense_1/bias:0 =  {dense_1_bias}')

        # 批量预测
        pre_id = sess.run(out_class_ids, feed_dict={sess.graph.get_tensor_by_name(in_tensor_name): eval_features})

        # 单样本预测 需保证feed的tensor各个维度的特征与train时的一致，这里不知道怎么像estimator那样输入时指定key???
        # 输入时每个维度对应的特征依次为 PetalLength PetalWidth SepalLength SepalWidth
        # pre_id = sess.run(out_class_ids, feed_dict={sess.graph.get_tensor_by_name(in_tensor_name): eval_features_0})


        # 值和特征不对应，导致准确率特别低
        # pre_id,pre_logits,pre_probabilities = sess.run([out_class_ids,out_logits,out_probabilities],
        #                               feed_dict={sess.graph.get_tensor_by_name(in_tensor_name): [[5.9,3.0,4.2,1.5]]})

        pre_id = np.array(pre_id).reshape(-1)
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

    print(type(args))
    print(args.mode)
    print(args.model_dir)

    if not args.mode or args.mode != 'train' or args.mode != 'eval':
        print('you must set the mode like this: --mode <train / eval>')

    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    if args.mode == 'train':
        train(train_x, train_y, DNNConfig.batch_size)

    if args.mode == 'eval' and not args.model_dir:
        print('you must set the model_dir like this: --model_dir <correct model path>')

    if args.mode == 'eval' and args.model_dir:
        eval(test_x, test_y, DNNConfig.batch_size, args.model_dir)
