import tensorflow as tf
import os
import argparse

ckpt_path = 'my_saver/saver.ckpt'

freeze_dir = 'my_freeze'

freeze_model_path = os.path.join(freeze_dir, 'frozen_model.pb')

parser = argparse.ArgumentParser()

parser.add_argument('--mode', default=None, type=str,
                    help='freeze or eval mode')


def freeze_and_save_model():
    saver = tf.train.import_meta_graph(ckpt_path + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    print("%d ops in the input_graph_def." % len(input_graph_def.node))
    print([node.name for node in input_graph_def.node])

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        output_node_name = ['output/predict_result/predicted_classes']
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                                        input_graph_def=input_graph_def,
                                                                        output_node_names=output_node_name)
        with tf.gfile.GFile(freeze_model_path, 'wb')as f:
            f.write(output_graph_def.SerializeToString())

        print("%d ops in the output_graph_def." % len(output_graph_def.node))
        print([node.name for node in output_graph_def.node])


def load_frozen_model():
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(freeze_model_path, "rb") as f:
            print(f'load frozen graph from {freeze_model_path}')
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
        with tf.Session() as sess:
            input_tensor = sess.graph.get_tensor_by_name('input_features:0')
            output_tensor = sess.graph.get_tensor_by_name('output/predict_result/predicted_classes:0')
            pre_id = sess.run(output_tensor, feed_dict={input_tensor: [[1.7,0.5,5.1,3.3]]})
            print(f'pre_id:{pre_id}')


if __name__ == '__main__':
    args = parser.parse_args()

    if not args.mode or args.mode != 'freeze' or args.mode != 'eval':
        print('you must set the mode like this: --mode <freeze / eval>')

    if args.mode == 'freeze':
        freeze_and_save_model()

    if args.mode == 'eval':
        load_frozen_model()
