import random
from preprocess import preprocess_text as preprocess


with open('../../data/filt.txt', 'r', encoding='utf-8') as filt:
    filt_lines = filt.readlines()

with open('../../data/ori.txt', 'r', encoding='utf-8') as ori:
    ori_lines = ori.readlines()

cate_dic = {'ori':1, 'filter':2}

sentences = []

preprocess(filt_lines, sentences, cate_dic['filter'])
preprocess(ori_lines, sentences, cate_dic['ori'])
# 随机打乱数据
random.shuffle(sentences)

# 将数据保存到train_data.txt中
print("writing data to fasttext format...")
out = open('../../data/train/train_data.txt', 'w', encoding='utf-8')
for sentence in sentences:
    out.write(sentence+"\n")
print("Saving train_data done!")
