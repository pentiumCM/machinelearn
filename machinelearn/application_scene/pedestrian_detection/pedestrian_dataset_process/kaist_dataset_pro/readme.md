# KAIST 数据集
## 一、 数据集预处理

## 1. 数据集图片处理
KAIST数据集中每张图片由一张可见光和一张红外图片组成，涉及两个模态的图片数据。

所以一个场景的图片分为两张图片，对应的文件名规则为：
    set_V_图片序号_模态，如：
    
    set00_V000_I00000_visible.jpg
    set00_V000_I00000_lwir.jpg
    
 

### 2. vbb 标签文件转 xml

+ 标注文件介绍

官方提供的标注文件如下：
```bash
├─annotations-vbb
│  ├─set00
│  ├─set01
│  ├─set02
│  ├─set03
│  ├─set04
│  ├─set05
│  ├─set06
│  ├─set07
│  ├─set08
│  ├─set09
│  ├─set10
│  └─set11
```
<br>

+ 处理规则

文件名：set_V_图片名，如：

set00_V000_I00980.xml
    