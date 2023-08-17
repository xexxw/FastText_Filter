import fasttext


# 读取模型
classifier = fasttext.load_model('../../checkpoints/fasttext/classifier_2023-08-17-18-47.model')


# 验证集准确率
result = classifier.test('../../data/val/val_data.txt')

print('P@1:', result[1])
print('R@1:', result[2])
print('Number of examples:', result[0])