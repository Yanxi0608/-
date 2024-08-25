#!/usr/bin/env python
# coding: utf-8

# In[5]:


#利用Apriori算法来发现频繁集

#先定义或加载DataSet      eg.DataSet = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#构建第一个候选集的列表C1
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return list(map(frozenset, C1))

#计算第k个候选集Ck的支持度，同时返回已经过支持度值域过滤的频繁项集Lk
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

#apriorGen函数用于创建候选集Ck
def aprioriGen(Lk, k): 
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): #k-2操作是为了减少重复操作
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: 
                retList.append(Lk[i] | Lk[j]) 
    return retList

#apriori算法（利用前面三个辅助函数进行！）
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k) #生成候选集ck
        Lk, supK = scanD(D, Ck, minSupport) #基于候选集与支持度值域生成频繁项集Lk
        supportData.update(supK) #更新支持度
        L.append(Lk)
        k += 1
    return L, supportData

L,suppData = apriori(DataSet)
#可用print返回L与suppData


#利用Apriori算法来挖掘关联规则
#提取关联规则并计算置信度
def generateRules(L, supportData, minConf=0.7):  
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

#计算单个规则的置信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] 
    for conseq in H:
        #置信度函数：con(A --> B) = support(A|B) / support(A)
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

#递归生成规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): 
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
            
rules = generateRules(L,suppData,0.7)
rules


# In[11]:


#案例：发现毒蘑菇的相似特征
#？？如何将数据以DataSet的形式加载？
file_path = r'C:\Users\86135\Desktop\machinelearninginaction\Ch11\mushroom.dat'
DataSet = [line.split() for line in open(file_path).readlines()]

#利用Apriori算法来发现频繁集

#先定义或加载DataSet      eg.DataSet = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

#构建第一个候选集的列表C1
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
                
    C1.sort()
    return list(map(frozenset, C1))

#计算第k个候选集Ck的支持度，同时返回已经过支持度值域过滤的频繁项集Lk
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt: ssCnt[can]=1
                else: ssCnt[can] += 1
    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0,key)
        supportData[key] = support
    return retList, supportData

#apriorGen函数用于创建候选集Ck
def aprioriGen(Lk, k): 
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk): #k-2操作是为了减少重复操作
            L1 = list(Lk[i])[:k-2]; L2 = list(Lk[j])[:k-2]
            L1.sort(); L2.sort()
            if L1==L2: 
                retList.append(Lk[i] | Lk[j]) 
    return retList

#apriori算法（利用前面三个辅助函数进行！）
def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = list(map(set, dataSet))
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k) #生成候选集ck
        Lk, supK = scanD(D, Ck, minSupport) #基于候选集与支持度值域生成频繁项集Lk
        supportData.update(supK) #更新支持度
        L.append(Lk)
        k += 1
    return L, supportData

L,suppData = apriori(DataSet)
#可用print返回L与suppData


#利用Apriori算法来挖掘关联规则
#提取关联规则并计算置信度
def generateRules(L, supportData, minConf=0.7):  
    bigRuleList = []
    for i in range(1, len(L)):
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList         

#计算单个规则的置信度
def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    prunedH = [] 
    for conseq in H:
        #置信度函数：con(A --> B) = support(A|B) / support(A)
        conf = supportData[freqSet]/supportData[freqSet-conseq]
        if conf >= minConf: 
            print (freqSet-conseq,'-->',conseq,'conf:',conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

#递归生成规则
def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): 
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        if (len(Hmp1) > 1):    
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)

#可限定范围！寻找包含特征值2的频繁项集
L,suppData = apriori(DataSet,0.3)
for item in L[1]:
    if item.intersection('2'):
        print (item)


# In[12]:


#FP-growth算法
#构建FP树

#对DataSet进行格式化处理
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

#FP树的类定义
class treeNode:
    #初始化节点属性
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode      
        self.children = {} 
    
    #增加节点计数
    def inc(self, numOccur):
        self.count += numOccur
    
    #显示树结构
    def disp(self, ind=1):
        print ('  '*ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind+1)

#FP树构建函数
def createTree(dataSet, minSup=1): 
    headerTable = {}
    #第一次遍历：计算各个项的支持度并移除不满足最小支持度的元素
    for trans in dataSet:  #计算支持度
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    keysToRemove = []
    for k in headerTable.keys():
        if headerTable[k] < minSup:
            keysToRemove.append(k)
    for k in keysToRemove:
        del(headerTable[k])
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0: return None, None  
    for k in headerTable:
        headerTable[k] = [headerTable[k], None] 
    retTree = treeNode('Null Set', 1, None) 
    
    #第二次遍历：建立树结构
    for tranSet, count in dataSet.items():  
        localD = {}
        for item in tranSet:  
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)]
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable 

#递归地将频繁项集插入或更新到树中
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count) 
    else:  
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        
#更新头指针表的链表，确保多个相同项的节点能够正确链接
def updateHeader(nodeToTest, targetNode):   
    while (nodeToTest.nodeLink != None):    
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode
    

#从叶节点向上追溯到根节点来构建路径(用递归调用的方式实现)
def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

#查找给定项的所有前缀路径
def findPrefixPath(basePat, treeNode): 
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1: 
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


# In[ ]:




