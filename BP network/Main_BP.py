from BP import *

del rawdata_1, rawdata_2, labelnum_1, labelraw_1, labelnum_2, labelraw_2  # 清空变量，释放内存
gc.collect()

# ################ rawdata_1 ########################
# ----------  裁剪+卷积算法  ------------
# X_cut = cutimage(X_1)  # 截取图像中嘴部的像素，尺寸为40×40
# convdata = extraimage(X_1, 2)  # 对图像进行简单卷积操作
# X_train, X_test, y_train, y_test, X_all_std = processdata(convdata, y_1, n_components=0, usepca=0,
#                                                           usehog=0)  # 分割和测试集，不使用pca和hog
# # trainBPmodel(size1, size2, max_iter, learning_rate, X_train, X_test, y_train, y_test, X_all_std)
# trainBPmodel(2, 3, 112, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
# 输出结果：
# spend time: 0:00:00.737491s      bp trainAcc:  0.8775510204081632
# bp testAcc: 0.837228714524207    cross_val_score Acc:  0.8211189331708552

# # ----------  原始数据直接拟合  ------------
# X_train, X_test, y_train, y_test, X_all_std = processdata(X_1, y_1, n_components=0, usepca=0,
#                                                           usehog=0)  # 不使用pca和hog，只对数据进行标准化
# trainBPmodel(2, 7, 100, 1e-4, X_train, X_test, y_train, y_test, X_all_std)
# # 输出结果：
# # spend time: 0:00:16.564327 s       bp trainAcc:  0.9316147511636234
# # bp testAcc:  0.7504173622704507    cross_val_score Acc:  0.7258706842199366

# ----------  HOG特征  ------------
# X_train, X_test, y_train, y_test, X_all_std = processdata(X_1, y_1, n_components=256, usepca=0, usehog=1)  # 使用hog
# trainBPmodel(2, 7, 100, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
# 输出结果：
# spend time:  0:00:20.490420 s       bp trainAcc:  0.9634801288936627
# bp testAcc: 0.7754590984974958    cross_val_score Acc:  0.7406554559113678

# ----------  LBP特征  -----------
# lbpdata = use_lbp(X)  # 提取lbp特征
# X_train, X_test, y_train, y_test, X_all_std = processdata(lbpdata, y, n_components=0, usepca=0, usehog=0)  # 不使用pca和hog
# trainBPmodel(2, 7, 100, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
# 输出结果：
# spend time: 0:00:20.817014 s        bp trainAcc:  0.9301825993555317
# bp testAcc: 0.7762938230383973     cross_val_score Acc:  0.6882851684906884

# ----------  PCA降维  ------------
X_train, X_test, y_train, y_test, X_all_std = processdata(X_1, y_1, n_components=256, usepca=1, usehog=0)  # 使用pca
trainBPmodel(2, 7, 100, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
# 输出结果：
# spend time: 0:00:00.488206 s       bp trainAcc:  0.9273182957393483
# bp testAcc: 0.7629382303839732    cross_val_score Acc:  0.7338838334886025

# ----------  LBP+PCA  ------------
# lbpdata = use_lbp(X_1)  # 提取lbp特征
# X_train, X_test, y_train, y_test, X_all_std = processdata(lbpdata, y_1, n_components=256, usepca=1, usehog=0)  # 使用pca和lbp
# trainBPmodel(2, 7, 100, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
# 输出结果：
# spend time: 0:00:00.595984 s     bp trainAcc:  0.8990332975295381
# bp testAcc: 0.7654424040066778    cross_val_score Acc:  0.7110796390224623

# ----------  HOG+PCA  ------------
# X_train, X_test, y_train, y_test, X_all_std = processdata(X_1, y_1, n_components=256, usepca=1, usehog=1)  # 使用pca和hog
# trainBPmodel(2, 7, 100, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
# 输出结果：
# spend time: 0:00:00.400830 s       bp trainAcc:  0.9308986752595775
# bp testAcc: 0.8130217028380634    cross_val_score Acc:  0.776242859965935



# ################ rawdata_2 ########################
# ----------  原始数据直接拟合  ------------
# X_train, X_test, y_train, y_test, X_all_std = processdata(X_2, y_2, n_components=0, usepca=0,
#                                                           usehog=0)  # 不使用pca和hog，只对数据进行标准化
# trainBPmodel(2, 7, 100, 1e-3, X_train, X_test, y_train, y_test, X_all_std)
# 输出结果：
# spend time: 0:00:01.020104 s       bp trainAcc:  0.5079365079365079
# bp testAcc:  0.5076784101174345    cross_val_score Acc:  0.5078590785907859
