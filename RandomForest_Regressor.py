import numpy as np
import pandas as pd
from copy import deepcopy
import os

def powerset(iterable):
    from itertools import chain, combinations
    xs = list(iterable)
    return chain.from_iterable( combinations(xs,n) for n in range(len(xs)+1) )

def get_powerset(iterable, length):
    subsets=set(powerset(iterable))
    ss=set([frozenset(s) for s in subsets if len(s)>=1 and len(s)<=length])
    return ss

def generate_folds(X, Y, nfolds=4):
    classes = np.unique(Y)
    cvlist = []
    
    for cidx, c in enumerate(classes):
        idx = Y == c
        Yt = Y[idx]
        Xt = X[idx, :]
        nexamples = Xt.shape[0]
        ridx = np.arange(nexamples)
        np.random.shuffle(ridx)
        nexamples = nexamples / nfolds
        nexamples = int(nexamples)
        sridx = set(ridx)
        sidx = 0
        for k in range(nfolds):
            testidx = ridx[sidx:sidx + nexamples]
            trainidx = list(sridx.difference(testidx))
            sidx += nexamples
            if cidx == 0:
                cvlist.append([Xt[trainidx, :], Yt[trainidx], Xt[testidx, :], Yt[testidx]])
            else:
                cvlist[k][0] = np.vstack((cvlist[k][0], Xt[trainidx, :]))
                cvlist[k][1] = np.hstack((cvlist[k][1], Yt[trainidx]))
                cvlist[k][2] = np.vstack((cvlist[k][2], Xt[testidx, :]))
                cvlist[k][3] = np.hstack((cvlist[k][3], Yt[testidx]))
    return cvlist

def getSplits(categories):
    categories = set(categories)
    tsplits = get_powerset(categories,len(categories)-1)
    flist=[]
    for s in tsplits:
        if not s in flist:
            r = categories.difference(s)
            flist.append(s)
            flist.append(r)

    olist=[]
    for s in flist[::-1]:
        ilist=[]
        for k in s:
            ilist.append(k)
        olist.append(tuple(ilist))    
    return olist

class Node:
    def __init__(self,purity,variance,mean=-1,fidx=-1, split=-1):
        self.lchild,self.rchild = None,None     
        self.split = split
        self.variance = variance
        self.mean = mean #Leaf Nodes will have mean
        self.fidx = fidx
        self.purity = purity
        self.ftype = 'categorical' if type(self.split) in [tuple, str, np.string_] else 'continuous'

    def set_childs(self,lchild,rchild):
        self.lchild = lchild
        self.rchild = rchild

    def isleaf(self):
        if self.mean != None:
            return True
        else:
            return False

    def isless_than_eq(self, X):
        if self.ftype == 'categorical':
            if X[self.fidx] in self.split:
                return True
            else:
                return False
        else:
            if X[self.fidx] <= self.split:
                return True
            else:
                return False
    
class RandomForestRegressor:
    '''Implements the Decision Tree For Regression'''
    def __init__(self,Path = None):
        self.numTrees = 3
        self.tree = []   
        self.purity = 0.0
        self.exthreshold = 0
        self.maxdepth = 0
        self.path = Path

    def __init__(self, purityp=0.05, exthreshold=5, maxdepth=10, numTrees = 3,Path = None):  
        self.numTrees = numTrees      
        self.purity = purityp
        self.exthreshold = exthreshold
        self.maxdepth = maxdepth
        self.tree = []
        self.path = Path

    def fit(self, X, Y):
        for IdxTrees in range(0, self.numTrees):
            nexamples = X.shape[0]
            nexamples = int(nexamples*0.7)
            ridx = np.arange(X.shape[0])
            np.random.shuffle(ridx)
            ridx = ridx[:nexamples]
            NewX = X[ridx, :]
            NewY = Y[ridx]
            self.tree.append(self.build_tree(NewX, NewY, self.maxdepth))
        self.Save_Model(self.path)
    
    def Save_Model(self,Path = None):
        if Path == None:
            Path = os.getcwd()
        print("Model Saved at: ",Path)
        np.save(Path,self.tree)
        
    def build_tree(self, X, Y, depth):
        nexamples, nfeatures = X.shape
        
        VarianceOfY = np.var(Y)
        Purity = VarianceOfY

        if ((nexamples < self.exthreshold) or (Purity < self.purity) or (depth < 0)):
            node = Node(Purity,-1,np.mean(Y))
        else:
            BEST_SP,BEST_VAR,BEST_LD,BEST_RD = None,None,None,None
            SP,VR,LD,RD= None,None,None,None
            Threshold = float("inf")
            Best_Feature = -1

            Num_Random_Features_To_Process = np.random.randint(1, nfeatures)
            Random_Features = np.random.choice(nfeatures, Num_Random_Features_To_Process, replace=False)
            
            for FeatureIndex in Random_Features:
                if np.dtype(X[0,FeatureIndex]) in [tuple, str, np.string_]:
                    if len(np.unique(X[:,FeatureIndex])) == 1:
                        continue
                    else:
                        SP,VR,LD,RD = self.evaluate_categorical_attribute(deepcopy(X[:,FeatureIndex]), deepcopy(Y))
                else:
                    if len(np.unique(X[:,FeatureIndex])) == 1:
                        continue
                    else:
                        SP,VR,LD,RD = self.evaluate_numerical_attribute(deepcopy(X[:, FeatureIndex]), deepcopy(Y))
                
                if  VR != None and VR < Threshold:
                    Best_Feature = FeatureIndex
                    BEST_SP,BEST_VAR,BEST_LD,BEST_RD = SP,VR,LD,RD
                    Threshold = BEST_VAR
                    
            node = None
            if BEST_VAR == None: #Leaf Node
                node = Node(Purity,-1,np.mean(Y)) #Purity, Variance, Mean
            else: #Internal Node
                node = Node(Purity,BEST_VAR,None,Best_Feature,BEST_SP) #Purity, Variance, NO Mean cauz leaf node will have mean only, FeatureIndex, Split
                node.set_childs(self.build_tree(deepcopy(X[BEST_LD]), deepcopy(Y[BEST_LD]), depth - 1), self.build_tree(deepcopy(X[BEST_RD]), deepcopy(Y[BEST_RD]), depth - 1))
        return node
        
    def Test(self, X):
        if self.path == None and len(self.tree) == 0:
            print("No Model Found")
            return
        else:
            Pred = self.predict(X)
            return np.array(Pred)

    def evaluate_categorical_attribute(self, feat, Y):
            categories = set(feat)
            splits = getSplits(categories)
            Vaiance_Matrix = []

            Num_Of_Split_Points_To_Keep = np.random.randint(1,len(splits)//2)
            Random_Split_Points = []
            for i in range(Num_Of_Split_Points_To_Keep):
                Random_Index = np.arange(0,len(splits),2)[np.random.randint(0,len(splits)/2)]
                Random_Split_Points.append(splits[Random_Index])
                Random_Split_Points.append(splits[Random_Index+1])
            splits = Random_Split_Points

            for idx,LeftSplit in range(0,len(splits),2):
                RightSplit = LeftSplit + 1

                LeftY = Y[np.isin(feat, splits[LeftSplit])]
                RightY = Y[np.isin(feat, splits[RightSplit])]

                LeftVariance = np.var(LeftY)
                RightVariance = np.var(RightY)

                Vaiance_Matrix.append(LeftVariance+RightVariance)

            LeftSplits = splits[0::2]
            RightSplits = splits[1::2]
            BestSplitIndex = np.argmin(Vaiance_Matrix)
            BestSplit = LeftSplits[BestSplitIndex]
            LeftDataIndices = np.isin(feat, LeftSplits[BestSplitIndex])
            RightDataIndices = np.isin(feat, RightSplits[BestSplitIndex])
            
            return BestSplit, Vaiance_Matrix[BestSplitIndex], LeftDataIndices, RightDataIndices
        
    def evaluate_numerical_attribute(self, feat, Y):
        UniquesInF = np.unique(sorted(feat))
        SplitPoints = []
        for i in range(len(UniquesInF)-1):
            SplitPoints.append((UniquesInF[i] + UniquesInF[i+1])/2)

        if len(SplitPoints) > 2:
            Random_Split_Points_To_Keep = np.random.randint(2,len(SplitPoints))
            np.random.shuffle(SplitPoints)
            SplitPoints = SplitPoints[:Random_Split_Points_To_Keep]

        Vaiance_Matrix = []
        for SplitPoint in SplitPoints:
            LeftY = Y[feat <= SplitPoint]
            RightY = Y[feat > SplitPoint]

            Left_Variance = np.var(LeftY)
            Right_Variance = np.var(RightY)

            Vaiance_Matrix.append(Left_Variance+Right_Variance)
        
        BestSplitIndex = np.argmin(Vaiance_Matrix)
        BestSplit = SplitPoints[BestSplitIndex]
        LeftDataIndices = np.where(feat <= BestSplit)[0]
        RightDataIndices = np.where(feat > BestSplit)[0]
        return BestSplit, Vaiance_Matrix[BestSplitIndex], LeftDataIndices, RightDataIndices

    def predict(self, X):
        z = []
        for idx in range(X.shape[0]):
            Temp = []
            for tree in self.tree:
                Temp.append(self._predict(tree, X[idx, :]))
            z.append(np.mean(Temp))
        return z
    
    def _predict(self, node, X):
        TreeRootTemp = node
        while TreeRootTemp.isleaf() == False:
            if TreeRootTemp.isless_than_eq(X) == True:
                TreeRootTemp = TreeRootTemp.lchild
            else:
                TreeRootTemp = TreeRootTemp.rchild
        return TreeRootTemp.mean #The leaf node at which the new test example fits best will have the mean of the target variable as the prediction    
   
    def find_depth(self):
        return self._find_depth(self.tree)

    def _find_depth(self, node):
        if not node:
            return
        if node.isleaf():
            return 1
        else:
            return max(self._find_depth(node.lchild), self._find_depth(node.rchild)) + 1

