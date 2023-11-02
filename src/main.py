import fasttext
import datetime
import random
import argparse
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='../../data/multi-label/0721/train.txt', required=False, help='input file')
parser.add_argument('--test', type=str, default='../../data/multi-label/0721/test2.txt', required=False, help='test file')
parser.add_argument('--lr', type=float, default=0.25, required=False, help='learning rate')
parser.add_argument('--dim', type=int, default=200, required=False, help='dimension')
parser.add_argument('--ngrams', type=int, default=3, required=False, help='ngrams')
parser.add_argument('--epoch', type=int, default=20, required=False, help='epoch')
parser.add_argument('--loss', type=str, default='softmax', required=False, help='loss function')
parser.add_argument('--threshold', type=float, default=0.5, required=False, help='threshold')
parser.add_argument('--pretrainedVectors', type=str, default=None, required=False, help='pretrained vectors')
parser.add_argument('--multi_lable', type=bool, default=False, required=False, help='multi label')
args = parser.parse_args()

print(args)

"""
  训练一个监督模型, 返回一个模型对象

  @param input:           训练数据文件路径
  @param lr:              学习率
  @param dim:             向量维度
  @param ws:              cbow模型时使用
  @param epoch:           次数
  @param minCount:        词频阈值, 小于该值在初始化时会过滤掉
  @param minCountLabel:   类别阈值，类别小于该值初始化时会过滤掉
  @param minn:            构造subword时最小char个数
  @param maxn:            构造subword时最大char个数
  @param neg:             负采样
  @param wordNgrams:      n-gram个数
  @param loss:            损失函数类型, softmax, ns: 负采样, hs: 分层softmax
  @param bucket:          词扩充大小, [A, B]: A语料中包含的词向量, B不在语料中的词向量
  @param thread:          线程个数, 每个线程处理输入数据的一段, 0号线程负责loss输出
  @param lrUpdateRate:    学习率更新
  @param t:               负采样阈值
  @param label:           类别前缀
  @param pretrainedVectors: 预训练的词向量文件路径, 如果word出现在文件夹中初始化不再随机
  @return model object
"""
if args.pretrainedVectors:
    model = fasttext.train_supervised(input=args.input, 
                                  dim=args.dim, 
                                  epoch=args.epoch,
                                  lr=args.lr,
                                  wordNgrams=args.ngrams, 
                                  loss=args.loss,
                                  pretrainedVectors=args.pretrainedVectors)
else:
    model = fasttext.train_supervised(input=args.input, 
                                  dim=args.dim, 
                                  epoch=args.epoch,
                                  lr=args.lr,
                                  wordNgrams=args.ngrams, 
                                  loss=args.loss)

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
model_path = f'../../checkpoints/fasttext/classifier_{current_time}.model'
print('Save path:', model_path)
model.save_model(model_path)
# classifier = fasttext.load_model('../../checkpoints/fasttext/classifier.model')

# 训练集准确率
result = model.test(args.input)

print('P@1:', result[1])
print('R@1:', result[2])
print('Number of examples:', result[0])

# 验证集准确率
result = model.test(args.test)

print('P@1:', result[1])
print('R@1:', result[2])
print('Number of examples:', result[0])

# 读取验证集数据
with open(args.test, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    true_labels = [line.split()[0] for line in lines]

# 预测验证集标签
predicted_labels = model.predict([line.strip() for line in lines])[0]

# 计算精确率、召回率和F1分数
precision = precision_score(true_labels, predicted_labels, average='macro')
recall = recall_score(true_labels, predicted_labels, average='macro')
f1 = f1_score(true_labels, predicted_labels, average='macro')
classifier_report = classification_report(true_labels, predicted_labels, digits=4)

# 计算混淆矩阵
confusion_mat = confusion_matrix(true_labels, predicted_labels)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(confusion_mat)
print("Classification Report:")
print(classifier_report)
