import tensorflow as tf
import geo_data_prepare
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import geo_ESIM
import geo_config as config
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn import metrics
import os
import random
import logging


con = config.Config()
parent_path = os.path.dirname(os.getcwd())
data_pre = geo_data_prepare.Data_Prepare()


class TrainModel(object):
    # Convert text to index  把原始文件如 test.txt 转换为 3个文件： query1(把词转换为数字index)  query2   结果
    def pre_processing(self, input_name, vocab):
        input = input_name+".txt"
        output_texta = input_name + "_code_a.txt"
        output_textb = input_name + "_code_b.txt"
        output_lable = input_name + "_lable.txt"
        logging.info("input file %s , three output files: %s %s %s", input, output_texta, output_textb, output_lable)

        texta_index = []
        textb_index = []
        tag = []
        with open(input, 'r', encoding='UTF-8') as f:
            for line in f.readlines():
                line = line.strip().split("\t")
                #print(line)
                texta = data_pre.pre_processing(line[0])  # texta 为分词后的  词汇列表
                texta_index.append(data_pre.sentence2Index(texta, vocab))  # sentence2Index返回上述 词汇列表 转换成 word2vec词汇表的 index数字，为0代表不在词汇表中
                #print(texta, data_pre.sentence2Index(texta, vocab))
                textb = data_pre.pre_processing(line[1])
                textb_index.append(data_pre.sentence2Index(textb, vocab))
                #print(textb, data_pre.sentence2Index(textb, vocab))
                tag.append(line[2])
        # shuffle
        index = [x for x in range(len(texta_index))]
        random.shuffle(index)
        texta_new = [texta_index[x] for x in index]
        textb_new = [textb_index[x] for x in index]
        tag_new = [tag[x] for x in index]
        # save
        with open(output_texta, 'w', encoding='utf-8') as o1:
            for a in texta_new:
                o1.write(" ".join(str(i) for i in a)+'\n')
        with open(output_textb, 'w', encoding='utf-8') as o2:
            for b in textb_new:
                o2.write(" ".join(str(i) for i in b)+'\n')
        with open(output_lable, 'w', encoding='utf-8') as o3:
            for t in tag_new:
                o3.write(" ".join(str(i) for i in t)+'\n')

    # Get mini-batch
    def get_batches(self, texta, textb, tag):
        num_batch = int(len(texta) / con.Batch_Size)
        for i in range(num_batch):
            a = texta[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            b = textb[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            t = tag[i * con.Batch_Size:(i + 1) * con.Batch_Size]
            yield a, b, t

    # Get the length of sentence
    def get_length(self, trainX_batch):
        
        lengths = []
        for sample in trainX_batch:
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

    # Load pre-trained word embeddings
    def load_word2vec(self, filename):
        vocab = []
        embed = []
        cnt = 0
        fr = open(filename, 'r', encoding='UTF-8')

        # word2vec.bin 第一行是 词数*维度 统计信息，直接跳过
        line = fr.readline().strip()

        # print line
        word_dim = int(line.split(' ')[1])  # 第一行的第二个数据为vector维度  256维
        vocab.append("unk")  # 第一行特意增加了这个词，为啥？？
        embed.append([0] * word_dim)
        for line in fr:
            row = line.strip().split(' ')
            vocab.append(row[0])
            embed.append(row[1:])
        print("loaded word2vec")
        fr.close()
        return vocab, embed

    # Train the ESIM
    def trainModel(self):
        # Load pre-trained word embeddings
        with tf.name_scope("Embedding"):
            vocab, embed = self.load_word2vec("data/model/w2v/word_level/word2vec.bin")
            logging.info("--%s--%s--%s--",vocab[0],vocab[1],vocab[2])
            logging.info(embed[0])
            logging.info(embed[1])
            vocab_size = len(vocab)
            logging.info("vocab_size")
            logging.info(vocab_size)
            embedding_dim = len(embed[0])
            logging.info("embedding_dim")
            logging.info(embedding_dim)
            embedding = np.asarray(embed)

        # Load training/development datasets
        train_texta_index = data_pre.readfile('data/dataset/train_code_a.txt')
        train_texta_index = pad_sequences(train_texta_index, con.maxLen, padding='post')
        logging.info("train_code_a")
        logging.info(train_texta_index[0])
        logging.info(len(train_texta_index))
        train_textb_index = data_pre.readfile('data/dataset/train_code_b.txt')
        train_textb_index = pad_sequences(train_textb_index, con.maxLen, padding='post')
        logging.info("train_code_b")
        logging.info(train_textb_index[0])
        logging.info(len(train_textb_index))
        train_tag = data_pre.readfile('data/dataset/train_lable.txt')
        train_tag = self.to_categorical(np.asarray(train_tag, dtype='int32'))
        logging.info("train_lable")
        logging.info(train_tag[0])
        logging.info(len(train_tag))

        dev_texta_index = data_pre.readfile('data/dataset/dev_code_a.txt')
        dev_texta_index = pad_sequences(dev_texta_index, con.maxLen, padding='post')
        logging.info("dev_code_a")
        logging.info(dev_texta_index[0])
        logging.info(len(dev_texta_index))
        dev_textb_index = data_pre.readfile('data/dataset/dev_code_b.txt')
        dev_textb_index = pad_sequences(dev_textb_index, con.maxLen, padding='post')
        logging.info("dev_code_b")
        logging.info(dev_textb_index[0])
        logging.info(len(dev_textb_index))
        dev_tag = data_pre.readfile('data/dataset/dev_lable.txt')
        dev_tag = self.to_categorical(np.asarray(dev_tag, dtype='int32'))
        logging.info("dev_lable")
        logging.info(dev_tag[0])
        logging.info(len(dev_tag))

        # Shuffle training dataset
        index_1 = [x for x in range(len(train_texta_index))]
        random.shuffle(index_1)
        train_texta_new = [train_texta_index[x] for x in index_1]
        train_textb_new = [train_textb_index[x] for x in index_1]
        train_tag_new = [train_tag[x] for x in index_1]
        # Shuffle development dataset
        index_2 = [x for x in range(len(dev_texta_index))]
        random.shuffle(index_2)
        dev_texta_new = [dev_texta_index[x] for x in index_2]
        dev_textb_new = [dev_textb_index[x] for x in index_2]
        dev_tag_new = [dev_tag[x] for x in index_2]

        # Define model
        with tf.variable_scope('esim_model', reuse=None):
            # esim model
            model = geo_ESIM.ESIM(True, seq_length=len(train_texta_new[0]),
                                  class_num=len(train_tag_new[0]),
                                  vocabulary_size=vocab_size,
                                  embedding_dim=embedding_dim,
                                  embedding_matrix=embedding,
                                  hidden_num=con.hidden_num,
                                  l2_lambda=con.l2_lambda,
                                  learning_rate=con.learning_rate)

        # Train model
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            saver = tf.train.Saver()
            best_f1 = 0.0
            for time in range(con.epoch):
                logging.info("training " + str(time + 1) + ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
                model.is_trainning = True
                loss_all = []
                accuracy_all = []
                for texta, textb, tag in tqdm(
                        self.get_batches(train_texta_new, train_textb_new, train_tag_new)):
                    feed_dict = {
                        model.text_a: texta,
                        model.text_b: textb,
                        model.y: tag,
                        model.dropout_keep_prob: con.dropout_keep_prob,
                        model.a_length: self.get_length(texta),
                        model.b_length: self.get_length(textb)
                    }
                    _, cost, accuracy = sess.run([model.train_op, model.loss, model.accuracy], feed_dict)
                    loss_all.append(cost)
                    accuracy_all.append(accuracy)
                logging.info("Epoch:" + str((time + 1)) + "; training loss:" + str(np.mean(np.array(loss_all))) + "; accuracy: " +
                      str(np.mean(np.array(accuracy_all))))

                def dev_step():
                    loss_all_dev = []
                    accuracy_all_dev = []
                    predictions_dev = []
                    for texta, textb, tag in tqdm(
                            self.get_batches(dev_texta_new, dev_textb_new, dev_tag_new)):
                        feed_dict = {
                            model.text_a: texta,
                            model.text_b: textb,
                            model.y: tag,
                            model.dropout_keep_prob: 1.0,
                            model.a_length: np.array(self.get_length(texta)),
                            model.b_length: np.array(self.get_length(textb))
                        }
                        dev_cost, dev_accuracy, dev_prediction = sess.run([model.loss, model.accuracy,
                                                                       model.prediction], feed_dict)
                        loss_all_dev.append(dev_cost)
                        accuracy_all_dev.append(dev_accuracy)
                        predictions_dev.extend(dev_prediction)
                    y_true_dev = [np.nonzero(x)[0][0] for x in dev_tag_new]
                    logging.info(len(y_true_dev))
                    y_true_dev = y_true_dev[0:len(loss_all_dev) * con.Batch_Size]
                    f1 = f1_score(np.array(y_true_dev), np.array(predictions_dev), average='weighted')
                    logging.info('Outputs:\n')
                    logging.info(metrics.classification_report(np.array(y_true_dev), predictions_dev))
                    logging.info("Dev: loss {:g}, accuracy {:g}, f1 {:g}\n".format(np.mean(np.array(loss_all_dev)),
                                                                      np.mean(np.array(accuracy_all_dev)),f1))
                    return f1

                model.is_trainning = False
                f1 = dev_step()

                if f1 > best_f1:
                    best_f1 = f1
                    saver.save(sess, "data/model/esim/model.ckpt")
                    logging.info("Saved model success\n")


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    model = TrainModel()
    vocab, embed = model.load_word2vec("data/model/w2v/word_level/word2vec.bin")
    model.pre_processing("data/dataset/test", vocab)
    model.pre_processing("data/dataset/dev", vocab)
    model.pre_processing("data/dataset/train", vocab)
    model.trainModel()


# input_file = '/data/dataset/test.txt'
# output_a = '/data/dataset/test_code_a.txt'
# output_b = '/data/dataset/test_code_b.txt'
# output_l = '/data/dataset/test_lable.txt'
# vocab, embed = train.load_word2vec(
#     "/model/w2v/word level/word2vec.bin")
# logging.info(vocab[0],vocab[1],vocab[2])
# train.pre_processing(input_file, output_a, output_b, output_l, vocab)
