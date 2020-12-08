# -*-coding:cp936-*-

# 存在的问题：
# 1.无法输出为配对的模版字段和数据字段，也就无法验证是否有遗漏
# 2.对照表字典未必是最全的
# 现在的解决办法：
# 将配对的模版字段单独输出，并与原始模版字段人工比较

mbzdfileHandle = open("E:\\test_aj\\mbzd.txt", 'r')
sjzdfileHandle = open("E:\\test_aj\\sjzd.txt", 'r')
jg = open("E:\\test_aj\\jg.txt", 'w')
jgmb = open("E:\\test_aj\\jgmb.txt", 'w')
mbzdlist = mbzdfileHandle.readlines()  # 模版字段列表
sjzdlist = sjzdfileHandle.readlines()  # 将读到文件内容放到list中，数据字段列表

# dzbzd 对照表字典
dzbzd = {"1": "a", "2": "b", "3": "c"}
# 第一个for循环从：模版字段中取一个字段
# 第二个for循环：遍历对照表字典
# 第一个if判断：将从模版中取的字段与对照表字典中所有键进行比较，如果相等进入第三个for循环
# 第三个for循环：遍历数据字段
# 第二个if判断：如果数据字段中有字段等于前面符合条件的键对应的值
# 那就说明根据对照表字典，模版字段和数据字段可以配对
for line1 in mbzdlist:
    for key in dzbzd:
        if key == line1.strip():
            for line2 in sjzdlist:
                if line2.strip() == dzbzd[key]:
                    # print '%s = %s' %(key,dzbzd[key])
                    jg.write(key + "=" + dzbzd[key] + ",")
                    jgmb.write(key + "\n")

        else:
            continue

mbzdfileHandle.close()
sjzdfileHandle.close()
jg.close()
jgmb.close()
