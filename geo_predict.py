import logging

import tensorflow as tf
import numpy as np
import geo_data_prepare
from keras.preprocessing.sequence import pad_sequences
import geo_config as config
import random
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn import metrics
from geo_train import TrainModel

data_pre = geo_data_prepare.Data_Prepare()
con = config.Config()

class Predict(object):
    def __init__(self):
        self.vocab, self.embed = TrainModel().load_word2vec("data/model/w2v/word_level/word2vec.bin")

        # self.checkpoint_file = tf.train.latest_checkpoint('D:/Lydia/PycharmProjects/Deep learning for geocoding/model/esim')
        self.checkpoint_file = tf.train.latest_checkpoint('data/model/esim')
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
            self.sess = tf.Session(config=session_conf)
            with self.sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(self.checkpoint_file))
                saver.restore(self.sess, self.checkpoint_file)

                # Get the placeholders from the graph by name
                self.text_a = graph.get_operation_by_name("esim_model/text_a").outputs[0]
                self.text_b = graph.get_operation_by_name("esim_model/text_b").outputs[0]
                self.a_length = graph.get_operation_by_name("esim_model/a_length").outputs[0]
                self.b_length = graph.get_operation_by_name("esim_model/b_length").outputs[0]
                self.drop_keep_prob = graph.get_operation_by_name("esim_model/dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                self.prediction = graph.get_operation_by_name("esim_model/output/prediction").outputs[0]
                self.score = graph.get_operation_by_name("esim_model/output/score").outputs[0]
                self.loss = graph.get_operation_by_name("esim_model/output/score").outputs[0]

    def get_length(self, text):
        # sentence length
        lengths = []
        for sample in text:
            count = 0
            for index in sample:
                if index != 0:
                    count += 1
                else:
                    break
            lengths.append(count)
        return lengths

    def to_categorical(self, y, nb_classes=None):
        y = np.asarray(y, dtype='int32')
        if not nb_classes:
            nb_classes = np.max(y) + 1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y

    def get_batches(self, texta, textb, tag):
        num_batch = int(len(texta) / con.Batch_Size)
        for i in range(num_batch):
            a = texta[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            b = textb[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            t = tag[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            yield a, b, t

    def convert_sentence(self, sentence='宝安区福永街道107国道边机场综合楼1006'):
        tokens = data_pre.pre_processing(sentence)  # 分词
        token_indexes = data_pre.sentence2Index(tokens, self.vocab) # 词汇的index
        # batch = [token_indexes]  # 批量预测多组数据
        batch = [token_indexes, token_indexes, token_indexes]  # 批量预测多组数据
        after_pad = pad_sequences(batch, con.maxLen, padding='post')
        logging.info("convert_sentence: %s \n%s \n%s \n%s\n", sentence, str(tokens), str(token_indexes), str(after_pad))
        return np.array(after_pad)


    def predict(self, query_a='宝安区福永街道107国道边机场综合楼188006', query_b='盐田区渔民新村51号13212'):
        convert_a = self.convert_sentence(query_a)
        convert_b = self.convert_sentence(query_b)
        a_length = self.get_length(convert_a)
        b_length = self.get_length(convert_b)
        logging.info("len a: %s , len b: %s", a_length, b_length)

        feed_dict = {
            self.text_a: convert_a,
            self.text_b: convert_b,
            self.drop_keep_prob: 1.0,
            self.a_length: np.array(a_length),
            self.b_length: np.array(b_length)
        }

        y_pred = []

        y, s = self.sess.run([self.prediction, self.score], feed_dict) # score 为 [x,1-x], 如果x<0.5,预测y=0；否则y=1； x代表可信度
        y_pred.extend(y)

        logging.info("convert_a: %s\n y: %s, s: %s, y_pred: %s", str(convert_a),str(y), str(s), str(y_pred))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    predict = Predict()
    predict.predict()
    while True:
        query_a = input("Enter address 1: ")
        query_b = input("Enter address 2: ")
        predict.predict(query_a, query_b)
