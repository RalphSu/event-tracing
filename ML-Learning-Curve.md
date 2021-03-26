时序数据的异常检测阅读列表

## 1. 
 S-H-ESD 大体流程: https://www.cnblogs.com/en-heng/p/9202654.html - Done
 
 其中：Grubbs' Test的分布公式的原因需要进一步理解其理论依据：可以参考这里https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm
 
## 2.

 SHESD中用到典型的时序分解手法：STL，算法细节
 
 http://www.gardner.fyi/blog/STL-Part-I/ - Done
 
 http://www.gardner.fyi/blog/STL-Part-II/ 
 
 PartII 比较详细的介绍了算法过程，不过实际撸还是得参考代码： https://github.com/jrmontag/STLDecompose

 https://arxiv.org/pdf/2008.09245.pdf  Mediff的做法 - Done

## 3. RNN
 
 https://zybuluo.com/hanbingtao/note/541458
 
 https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
 
 https://eli.thegreenplace.net/2018/understanding-how-to-implement-a-character-based-rnn-language-model/
 

 
## 4. Autoencoder & VAE

https://anotherdatum.com/vae.html

https://anotherdatum.com/vae2.html

https://towardsdatascience.com/auto-encoder-what-is-it-and-what-is-it-used-for-part-1-3e5c6f017726

http://ufldl.stanford.edu/tutorial/unsupervised/Autoencoders/

https://zhuanlan.zhihu.com/p/27865705

https://medium.com/ai-academy-taiwan/what-are-autoencoders-175b474d74d1

https://medium.com/analytics-vidhya/an-introduction-to-generative-deep-learning-792e93d1c6d4

## 5. 
https://github.com/yzhao062/anomaly-detection-resources anomaly detection collections!

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152173 A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data


Unsupervised Anomaly Detection Approach for
Time-Series in Multi-Domains Using Deep
Reconstruction Error - 

## 7 Omni

https://www.kdd.org/kdd2019/accepted-papers/view/robust-anomaly-detection-for-multivariate-time-series-through-stochastic-re


## 6. Survey as overall introduction in this domain

ANOMALY DETECTION IN UNIVARIATE TIME-SERIES: A SURVEY ON THE STATE-OF-THE-ART - Should be a good read. - Done https://arxiv.org/pdf/2004.00433.pdf 把这个看完了，各种方法写的还挺齐全的。2020.4的论文，基本上的AD算法都cover了，可以按图索骥深入学习，比如iforest, xgboost

https://www.dfki.de/fileadmin/user_upload/import/10754_comparative_anomaly_detection_ICMLA.pdf : A Comparative Analysis of Traditional and Deep
Learning-based Anomaly Detection Methods for
Streaming Data

https://www.researchgate.net/publication/252671230_A_Comparative_Evaluation_of_Anomaly_Detection_Algorithms_for_Maritime_Video_Surveillance : A Comparative Evaluation of Anomaly Detection Algorithms for Maritime Video Surveillance

Book: OUTLIER ANALYSIS: Second Edition (2017)


## 7. ARIMA 
- TODO : an internal session

https://towardsdatascience.com/anomaly-detection-in-multivariate-time-series-with-var-2130f276e5e9?gi=7120f7e8e30a

### 8 . Outlier Detection in Sparse Data with Factorization Machines

http://mashuai.buaa.edu.cn/pubs/cikm2017b.pdf 

#### 2020 KDD


- Partial Multi-Label Learning via Probabilistic Graph Matching Mechanism

    Gengyu Lyu Songhe Feng Yidong Li 


- Isolation Distributional Kernel: A New Tool for Kernel based Anomaly Detection

    Kai Ming Ting Bi-Cun Xu Takashi Washio Zhi-Hua Zhou 


- Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks

    Zonghan Wu Shirui Pan Guodong Long Jing Jiang Xiaojun Chang Chengqi Zhang 

- Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns

    Qingsong Wen Zhe Zhang Yan Li Liang Sun 

- USAD: UnSupervised Anomaly Detection on Multivariate Time Series

    Julien Audibert Pietro Michiardi Frédéric Guyard Sébastien Marti Maria A. Zuluaga 

- Robust Deep Learning Methods for Anomaly Detection

    Raghavendra Chalapathy Nguyen Lu Dang Khoa Sanjay Chawla 

- Deep Learning for Anomaly Detection

    Ruoying Wang Kexin Nie Yen-Jung Chang Xinwei Gong Tie Wang Yang Yang Bo Long 
 
 
### How to build AI platform

https://medium.com/@louisdorard/an-overview-of-ml-development-platforms-df953060b9a9 这里有几个分类有点意思 - Done

- Semi-specialized platforms (e.g. for text or image inputs)

- High-level platforms as a service (mostly for tabular data)

- Self-hosted studios

- Cloud Machine Learning IDEs

https://medium.com/@louisdorard/architecture-of-a-real-world-machine-learning-system-795254bec646 --系列二 - Done




