from DataProcess import *

imgData = imgDataProcess()
labelData = labelDataProcess()
combineSeq(imgData, labelData)

# 生成'pdData.csv'
Data = pd.read_csv('pdData.csv', index_col=0)
print(list(set(Data['Face'])))

