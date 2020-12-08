# -*-coding:cp936-*-

# 删除相同的行

mbzdfileHandle = open("E:\\test_aj\\mbzd.txt", 'r')
jgmbfileHandle = open("E:\\test_aj\\jgmb.txt", 'r')

mbsy = open("E:\\test_aj\\mbsy.txt", 'w')  # 模版字段和模版结果的差集

mbzdlist = mbzdfileHandle.readlines()  # 模版字段列表
jgmblist = jgmbfileHandle.readlines()  # 将读到文件内容放到list中，数据字段列表

mbzdset = set(mbzdlist)  # 将列表转成集合
jgmbset = set(jgmblist)
cj = mbzdset - jgmbset  # 两个集合的差集
for i in cj:
    mbsy.write(i)  # 将差集写到文件中

mbzdfileHandle.close()
jgmbfileHandle.close()
mbsy.close()
