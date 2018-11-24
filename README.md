# HOG+SVM

使用说明：[点我查看博客](https://blog.csdn.net/q1242027878/article/details/74271694)

上面的hog_svm.py是用于训练的，通过提取图片的hog特征，然后通过SVM进行训练得到model，最后通过model进行预测,将结果写入result.txt文件之中。

代码不难，大家根据需要自己改改。

不要将hog的参数设置的太复杂。不然提取的特征会非常大，然后训练的时候会占满内存，导致机器死机。
