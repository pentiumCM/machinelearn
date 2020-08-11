# 预测股票走势
## 一、数据集
+ 参考资料
    + https://blog.csdn.net/hzk427/article/details/104074285

+ Quandl：
    + 网站：
    
        https://www.quandl.com/
    + 使用：
    
        pip install quandl
    + 属性：
        + 日期 (Date)
        + 开盘价 (Open)
        + 最高价 (High)
        + 最低价 (Low)
        + 收盘价 (Close)
        + 总交易额 (Volume)
        
<br>     
        
+ pandas-datareader模块获取股票数据:
    + pandas-datareader 模块介绍：
    
        Pandas库的子模块 pandas-datareader 提供了从 雅虎财经、新浪财经等网站获取股票等金融数据的接口。
        
    + 数据集股票代码：
    
        数据集名称为股票代码时，
        
        美股直接输入股票代码如 GOOG；
        
        其他股票输入格式为“股票代码.对应股市”，上交所股票后面加上.SS，深交所股票加上.SZ，港股加上.hk；
    
    + 使用：
        ```bash
        pip install pandas-datareader
        ```
        
        ```python
        import pandas_datareader.data as web
  
        #从雅虎财经读入2019年至今的上证指数数据
        df = web.DataReader("000001.SS","yahoo",date(2019, 1, 1),date.today())
        pd.set_option('display.max_columns',None)#显示所有列
        df.head()
        df.Close.plot()#画出上证指数收盘价走势
        ```
    + 属性：
        + 日期 (Date)
        + 最高价 (High)
        + 最低价 (Low)
        + 开盘价 (Open)
        + 收盘价 (Close)
        + 总交易额 (Volume)
        + 调整后的收盘价 (Adj Close)
        
