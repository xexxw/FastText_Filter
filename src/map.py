import sys
import fasttext
import json
import jieba
import re



# load model
model = fasttext.load_model('text_cls_fasttext.bin')

dict_cate = {'__label__1':'1', '__label__2':'2'}

# read content
for line in sys.stdin:
    line = line.strip()
    # 按制表符分割行
    columns = line.split('\t')
    # 输出第二列
    # 把json字符串转换成python对象
    data = json.loads(columns[1])
    content = data["content"]

    
    # preprocess data    
    content = content.lower()
    content = content.replace('[br]', '')
    # # 使用正则表达式过滤标点符号
    # content = re.sub(r'[^\w\s]', '', content)
    seg = jieba.lcut(content)
    content = " ".join(segs)
    
    
    # predict
    pred_label = model.predict(content)
    pred_res = str(dict_cate[pred_label[0][0]])

    
    # output
    if pred_res == '1':
        print('\t'.join(columns)+"#1")
    else:
        print('\t'.join(columns)+"#2")
