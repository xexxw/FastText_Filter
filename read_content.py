import json

txt_file = "../../data/ori_test.txt"
out_file = "../../data/ori.txt"

# 打开txt文件进行读取
with open(txt_file, "r", encoding="utf-8") as file:
    lines = file.readlines()

# 解析每行的JSON字符串并提取字段
data_list = []
for line in lines:
    data = json.loads(line)
    source = data["source"]
    content = data["content"]
    length = data["length"]
    filename = data["filename"]
    hashval = data["hashval"]
    data_list.append([content])


# # 将数据写入CSV文件
# headers = ["source", "content", "length", "filename", "hashval"]
data_lines = [item[0] for item in data_list]
with open(out_file, "w", encoding="utf-8") as file:
    file.write("\n".join(data_lines))



print("转换完成！")