# HOG+SVM

使用说明：[点我查看博客](https://blog.csdn.net/q1242027878/article/details/74271694)

上面的hog_svm.py是用于训练的，通过提取图片的hog特征，然后通过SVM进行训练得到model，最后通过model进行预测。

hog_svm_predict.py是用于使用上面得到的model进行图片的预测，会将解决写入文件之中。

代码不难，大家根据需要自己改改。