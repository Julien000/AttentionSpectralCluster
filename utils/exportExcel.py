from datetime import datetime
import logging
import xlwt
import csv
import os
# 仅适用于建立单个表格
def export_excel(sheet_name='sheet_name', 
                col = ('test1', 'test2'),
                datalist = [],
                savepath = './output_excel.xls'
                ):
    """
        该函数用于建立数据表格, \n
        sheet_name, 数据表名 有默认 sheet_Name\n
        col, 数据列名，  (tuple) \n
        datalist: 二维数组，数据按照行存储 \n
        savapath: 保存数据的路径 ，默认存储当前目录, 包含文件名 \n
        col = ('列名1', '列名2', '列名3')\n
        dalist= [ 
            [col1 , col2 , col3], 一定要采用[ [],]格式才行
            ...
        ]\n
        EXAMPLE :
        from export_excel import export_excel
        export_excel(col=(
            \n 'listName1', 'listName2 ', 'listName3'....),
            datalist=[
                [data1, data2 ,data3....],
                [data4,],....
            ],
        savepath = './execel/GAN_loss.xls'
            )
    """
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet("Sheet1",cell_overwrite_ok=True)
    # load column
    for i in range(0,len(col)):
        sheet.write(0,i, col[i])
    # loaddata
    for i in range(0, len(datalist)):
        data = datalist[i]
        for j in range(0, len(col)):
            sheet.write( i+1, j, str(data[j]))
    #save file
   
    savepath = savepath
    # os.makedirs(sheet_name+ '_' + str(datetime.now())[:10]+".xls")
    book.save(sheet_name+ '_' + str(datetime.now())[:10]+".xls")
   
def export_csv(path, rowList):
    '''
        path:  file path
        rowList: all rows data including the headcount
        [
            [row1, row2],
            ...
            [rown, rown],
        ]
        https://blog.csdn.net/allway2/article/details/118071806
    '''
    if not os.path.exists(path):
        # os.makedirs(path)
        file = open(path, 'w')
        file.close()

    with open(path, 'w', encoding='UTF-8',newline='') as file:
        writer = csv.writer(file)
        for i,data in enumerate(rowList) :
            writer.writerow(data)
        logging.info("export csv file succese")
    
 