class dataContainer():
    def __init__(self, epoch=1, len=12, round_size=5):
        """
            适用于聚类的结果容器方法。
        """
        self.round_size=round_size #去掉小数位数
        self.len = len
        self.epoch =epoch
        self.data = [0 for _ in range(len)]
        # sum+ ACC 0
        # sum +NMI 1
        # sum +ARI 2
        # sum+Purity 3
        # max+ACC 4
        # max+NMI 5
        # max +ARI 6
        # max+Purity  7
        # avgACC 8
        # avgNMI 9
        # avgARI 10
        # avgPurity 11
    def setdata(self, data):
        # [*(1,2,3,5)] 结果[1, 2, 3, 5] ;[a,b,c,d] =data
        # 得到的结果应该是 ACC , NMI ,ARI, Purity
        [ACC, NMI, ARI, P] = data
        # 前四个值是 累计的 精度值
        self.data[0] += ACC
        self.data[1] += NMI
        self.data[2] += ARI
        self.data[3] += P
    def GetOneEpochResult(self,size):
        ACC= round(self.data[0]/size, self.round_size)
        NMI= round(self.data[1]/size,self.round_size)
        ARI= round(self.data[2]/size,self.round_size)
        P= round(self.data[3]/size,self.round_size)
        # 更新最大的结果
        if self.data[4]<ACC : self.data[4] = ACC
        if self.data[5]<NMI : self.data[5] = NMI
        if self.data[6]<ARI : self.data[6] = ARI
        if self.data[7]<P : self.data[7] = P

        #TODO:# 定值的长度不太可取，但是没办法,
        self.data[8] += ACC
        self.data[9] += NMI
        self.data[10] += ARI
        self.data[11] += P

        # 重置 累计数据
        self.data[0] = 0
        self.data[1] = 0
        self.data[2] = 0
        self.data[3] = 0

        return [ACC,NMI,ARI,P]
    def ResetALL(self):
        self.data = [0 for _ in range(self.len)]
    def getMax(self):
        ACC =self.data[4] 
        NMI =self.data[5] 
        ARI =self.data[6] 
        P =self.data[7] 
        return [ACC, NMI, ARI, P]
    def getAvg(self, times):
        avg_ACC= round(self.data[8]/times, self.round_size)
        avg_NMI= round(self.data[9]/times,self.round_size)
        avg_ARI= round(self.data[10]/times,self.round_size)
        avg_P= round(self.data[11]/times,self.round_size)
        return [avg_ACC, avg_NMI, avg_ARI, avg_P]
    def getdata(self):

        return round(self.data[0],self.round_size),round(self.data[1],self.round_size),round(self.data[2],self.round_size),round(self.data[3],self.round_size)

class trainDataContainer():
    def __init__(self, epoch=1, len=12, round_size=3):
        """
            适用于聚类的结果容器方法。
        """
        self.round_size=round_size #去掉小数位数
        self.len = len
        self.epoch =epoch
        self.data = [0 for _ in range(len)]

    def setdata(self, data):
        # [*(1,2,3,5)] 结果[1, 2, 3, 5] ;[a,b,c,d] =data
        # 得到的结果应该是 ACC , NMI ,ARI, Purity
        [ACC, NMI, ARI, P] = data
        # 前四个值是 累计的 精度值
        self.data[0] += ACC
        self.data[1] += NMI
        self.data[2] += ARI
        self.data[3] += P
    def ResetALL(self):
        self.data = [0 for _ in range(self.len)]
    def getdata(self,size):

        ACC= round(self.data[0]/size, self.round_size)
        NMI= round(self.data[1]/size,self.round_size)
        ARI= round(self.data[2]/size,self.round_size)
        P= round(self.data[3]/size,self.round_size)

        return [ACC,NMI,ARI,P]