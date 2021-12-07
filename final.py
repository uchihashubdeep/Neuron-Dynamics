import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('times1.txt',
                   sep=";|:|,",
                   header=None,
                   engine='python')

data1 = pd.read_csv('indx.txt',
                   sep=";|:|,",
                   header=None,
                   engine='python')

data2 = pd.read_csv('size.txt',
                   sep=";|:|,",
                   header=None,
                   engine='python')

def CI(size,times,synapse):
    a4=[]
    for i in range(len(times)):
        for j in range(int(size[i])):
            a4.append(times[i])
    bins = np.arange(0, max(a4), 1.6)
    digitized = np.digitize(a4, bins)
    no=np.zeros(len(bins),dtype='int')
    i=0
    for i in range(len(digitized)):
        no[digitized[i]-1]=no[digitized[i]-1]+1
    s=0    
    for i in range(len(no)):
        s=s+no[i]
    x=[]
    k=0
    for i in range(len(no)):
        micro=np.arange(i*1.6,(i+1)*1.6,0.01)
        l=[]
        for j in range(no[i]):
            l.append(a4[k])
            k=k+1
        ci=np.digitize(l, micro)
        uni=list(set(ci))
        x.append(len(uni)/160)
    cluster_index=x[-1]
    y=np.arange(1,max(digitized)+1,1)
    plt.plot(y,x)
    return cluster_index,y,x

def auxilliary(data,data1,data2,k):
    times=[]
    i=0
    j=k
    while(not np.isnan(data[j][0]) and data[j][0] != 0 and j<k+100000):
        times.append(data[j][0])
        i=i+1
        j=j+1
        if(i>99998):
            print(i)
    x=len(times)
    indx=[]
    i=0
    j=k
    while(i<x):
        indx.append(data1[j][0])
        i=i+1
        j=j+1
    indx1=[]
    i=x-1
    while(indx[i] == 0.0):
        i=i-1
    indx1=indx[0:i+1]
    size=[]
    i=0
    while(data2[i+k][0] != 0 and i+k<k+100000):
        size.append(data2[i+k][0])
        i=i+1
    return times,size,indx1

j=0
y1=[]
x1=[]
ci=[]
syn=[]
k=0
n1=10001
while(j<n1):
    n=100000*j
    times2,size2,indx3=auxilliary(data,data1,data2,n)
    print("CI comp started")
    ci3,y,x=CI(size2,times2,k)
    y1.append(y)
    x1.append(x)
    print(ci3)
    ci.append(ci3)
    syn.append(j)
    print(str(j)+" synapse removed")
    j=j+1
    k=k+1
    
plt.plot(syn,ci)
    
    
