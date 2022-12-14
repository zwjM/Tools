import random
import datetime
from sqlite3 import connect
import pandas as pd
import pymysql
import pymssql
class Sqlserver:
    #对调用数据库的方法进行封装
    def __init__(self,host,user,password,database):
        self.host=host #主机名
        self.user=user #用户名
        self.password=password #登陆密码
        self.database=database #数据库名


    def getCur(self):
        # 连接数据库
        if not self.database:
            raise(NameError,"数据库名称错误")#异常处理
        #连接数据库
        self.connect=pymysql.connect(host=self.host,user=self.user,password=self.password,database=self.database,cursorclass=pymysql.cursors.Cursor)
        # self.connect=pymssql.connect(host=self.host,user=self.user,password=self.password,database=self.database)
        cur=self.connect.cursor()#设个游标

        if not cur:
            raise(NameError,"数据库连接失败")
        else:
            return cur
        
    def CreateTable(self,sql):
        #创建数据表
        cur=self.getCur()
        cur.executemany(sql)
    #删除语句
    def delete(self,sql):
        try:
            cur = self.getCur()
            cur.execute(sql)
            self.connect.commit()
            print('finish')
        except Exception as e:
            print('delete error:'+e)
        finally:
            self.connect.close()
            cur.close()


    def select(self,sql,showall=True):
        #查询语句
        cur=self.getCur()
        timeQ1=datetime.datetime.now()
        num = cur.execute(sql) 
        print('一共：%d条'%num)
        data=cur.fetchall() #一次性取出所有数据
        
        if showall:
            print(pd.DataFrame(data))
        self.connect.close() #查询完毕后必须关闭连接
        timeQ2=datetime.datetime.now()
        t_sel=timeQ2-timeQ1 # 查询所用的时间
        if not sql:
            raise(NameError,"没有Sql语句")
        else:
            print('用时：',t_sel)
        #return t_sel
    def InsertR(self,sql,info):
        #插入多条数据
        cur=self.getCur()
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

class msSqlSeverer:
    #对调用数据库的方法进行封装
    def __init__(self,host,user,password,database):
        self.host=host #主机名
        self.user=user #用户名
        self.password=password #登陆密码
        self.database=database #数据库名
    def getCur(self):
        self.connect = pymssql.connect(host =self.host,user =self.user,password = self.password,database = self.database)
        cur=self.connect.cursor()#设个游标
        if not cur:
            raise(NameError,"数据库连接失败")
        else:
            return cur
    def create_table(self,sql):
        cur = self.getCur()
        cur.executemany(sql)
    def select(self,sql,showall=True):
        #查询语句
        cur=self.getCur()
        timeQ1=datetime.datetime.now()
        num = cur.execute(sql) 
        print('一共：%d条'%num)
        data=cur.fetchall() #一次性取出所有数据
        
        if showall:
            print(pd.DataFrame(data))

        
        self.connect.close() #查询完毕后必须关闭连接
        timeQ2=datetime.datetime.now()
        t_sel=timeQ2-timeQ1 # 查询所用的时间
        if not sql:
            raise(NameError,"没有Sql语句")
        else:
            return t_sel
    def insectR(self,sql,info):
        
        try:
            cur =self.getCur()
            cur.executemany(sql,info)
            self.connect.commit()
            print("finish")
        except Exception as e:
            print("insert error:"+e)
        finally:
            cur.close()
    def delete(self,sql):
        try:
            cur = self.getCur()
            cur.execute(sql)
            self.connect.commit()
            print('finish')
        except Exception as e:
            print('delete error:'+e)
        finally:
            self.connect.close()
            cur.close()


