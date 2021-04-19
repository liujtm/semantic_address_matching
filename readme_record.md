操作记录：
1. 从文本生成词 geo_token.py  (缺失分词词库文件GeoDicv2.txt和Stopwords.txt)
2. 训练生成词向量文件 word2vec.bin（有两种方法：geo_word2vec.py 和 CRF的 other_crf_w2v.py）
3. 训练模型 geo_train.py 
   - 把原始文件如 test.txt 转换为 3个文件： query1(把词转换为数字index)  query2 结果
   - 首先加载上述词向量word2vec.bin;
   - 再加载训练集和dev开发集各三个文件(label code_a code_b)
   - 训练，输出模型文件 data/model/esim/model.ckpt , 训练20轮 epoch 再MBP上耗时接近2小时；
4. 测试  geo_test.py
