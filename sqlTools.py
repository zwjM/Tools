import random
import datetime
import pymysql
class Sqlserver:
    #对调用数据库的方法进行封装
    def __init__(self,host,user,password,database):
        self.host=host #主机名
        self.user=user #用户名
        self.password=password #登陆密码
        self.database=database #数据库名


    def Getconnect(self):
        # 连接数据库
        if not self.database:
            raise(NameError,"数据库名称错误")#异常处理
        #连接数据库
        self.connect=pymysql.connect(host=self.host,user=self.user,password=self.password,database=self.database,cursorclass=pymysql.cursors.Cursor)
        # self.connect=pymssql.connect(host=self.host,user=self.user,password=self.password,database=self.database)
        cur=self.connect.cursor()#设个游标

        print(type(self.connect))
        print(type(cur))
        if not cur:
            raise(NameError,"数据库连接失败")
        else:
            return cur
        
    '''def CreateTable(self,sql):
        #创建数据表
        cur=self.Getconnect()
        cur.executemany(sql)'''

    def Query(self,sql):
        #查询语句
        cur=self.Getconnect()
        timeQ1=datetime.datetime.now()
        for j in range(1,101):
            cur.execute(sql)
        #data=cur.fetchall() #一次性取出所有数据
        self.connect.close() #查询完毕后必须关闭连接
        timeQ2=datetime.datetime.now()
        t_sel=timeQ2-timeQ1 # 查询所用的时间
        if not sql:
            raise(NameError,"没有Sql语句")
        else:
            return t_sel
        #return t_sel
    def InsertR(self,sql,num,info):
        #插入多条数据
        cur=self.Getconnect()
        timeI1=datetime.datetime.now()
        cur.executemany(sql,info)
        self.connect.commit()
        cur.close()
        self.connect.close()
        timeI2=datetime.datetime.now()
        t_ins=timeI2-timeI1 # 插入所用的时间
        if not sql:
            raise(NameError,"没有Sql语句")
        else:
            return t_ins


'''使用说明
    #登陆，连接数据库
    num=eval(input('请输入所要输入数据的条数：'))
    ss=Sqlserver(host="MSI",user="sa",password="qwerty",database="test")
    #create_sql="creat table T1 (id int,L1 varchar(2),L2 varchar(2),L3 varchar(2),L4 varchar(2),L5 varchar(2),L6 varchar(2),L7 varchar(2),L8 varchar(2),L9 varchar(2))"
    #ss.CreateTable(create_sql)

    #insert_sql = "insert into T1 values(%d,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    #insert_sql = "insert into T2 values(%d,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
    insert_sql = "insert into T3 values(%d,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

    ran = random.choice("abcdefghmklwpqxy")

    info = [(i, "w%s"%(i), ran, ran, ran, ran, ran, ran, ran, ran) for i in range(1,num+1)]

    #Query_sql="select top (10000) * from T1"#查询前10000条数据
    #Query_sql = "select top (10000) * from T2"  # 查询前10000条数据
    Query_sql = "select top (10000) * from T3"  # 查询前10000条数据

    t_ins=ss.InsertR(insert_sql,num,info)

    t_sel=ss.Query(Query_sql)
    print("插入",num,"条数据所用的时间",t_ins)
    print("连续查询100次10000条数据所用的时间",t_sel)'''
