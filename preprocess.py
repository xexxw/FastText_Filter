import jieba
import re


def preprocess_text(content_lines, sentences, category):
    for line in content_lines:
        try:
            line = line.lower()
            line = line.replace('[br]', '')
            segs = jieba.lcut(line)
            # 使用正则表达式过滤标点符号
            line = re.sub(r'[^\w\s]', '', line)
            
            # 过滤长度小于1的词 
            segs = list(filter(lambda x:len(x)>1, segs))

            # 将句子处理成  __label__1 词语 词语 词语 ……的形式
            sentences.append("__label__"+str(category)+" , "+" ".join(segs))
        except Exception as e:
            print(Exception, e)
            continue
