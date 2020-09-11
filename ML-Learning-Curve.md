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
 

 
## 4. VAE 

https://anotherdatum.com/vae.html

https://anotherdatum.com/vae2.html


## 5. 
https://github.com/yzhao062/anomaly-detection-resources anomaly detection collections!

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0152173 A Comparative Evaluation of Unsupervised Anomaly Detection Algorithms for Multivariate Data


Unsupervised Anomaly Detection Approach for
Time-Series in Multi-Domains Using Deep
Reconstruction Error - 

## 6. Survey as overall introduction in this domain

ANOMALY DETECTION IN UNIVARIATE TIME-SERIES: A SURVEY ON THE STATE-OF-THE-ART - Should be a good read.

https://www.dfki.de/fileadmin/user_upload/import/10754_comparative_anomaly_detection_ICMLA.pdf : A Comparative Analysis of Traditional and Deep
Learning-based Anomaly Detection Methods for
Streaming Data

https://www.researchgate.net/publication/252671230_A_Comparative_Evaluation_of_Anomaly_Detection_Algorithms_for_Maritime_Video_Surveillance : A Comparative Evaluation of Anomaly Detection Algorithms for Maritime Video Surveillance


#### 2020 KDD


- Partial Multi-Label Learning via Probabilistic Graph Matching Mechanism

    Gengyu Lyu Songhe Feng Yidong Li 

Partial Multi-Label learning (PML) learns from the ambiguous data where each instance is associated with a candidate label set, where only a part is correct. The key to solve such problem is to disambiguate the candidate label sets and identify the correct assignments between instances and their ground-truth labels. In this paper, we interpret such assignments as instance-to-label matchings, and formulate the task of PML as a matching selection problem. To model such problem, we propose a novel grapH mAtching based partial muLti-label lEarning (HALE) framework, where Graph Matching scheme is incorporated owing to its good performance of exploiting the instance and label relationship. Meanwhile, since conventional one-to-one graph matching algorithm does not satisfy the constraint of PML problem that multiple instances may correspond to multiple labels, we extend the traditional probabilistic graph matching algorithm from one-to-one constraint to many-to-many constraint, and make the proposed framework to accommodate to the PML problem. Moreover, to improve the performance of predictive model, both the minimum error reconstruction and k-nearest-neighbor weight voting scheme are employed to assign more accurate labels for unseen instances. Extensive experiments on various data sets demonstrate the superiority of our proposed method.


- Isolation Distributional Kernel: A New Tool for Kernel based Anomaly Detection

    Kai Ming Ting Bi-Cun Xu Takashi Washio Zhi-Hua Zhou 

We introduce Isolation Distributional Kernel as a new way to measure the similarity between two distributions. Existing approaches based on kernel mean embedding, which converts a point kernel to a distributional kernel, have two key issues: the point kernel employed has a feature map with intractable dimensionality; and it is data independent. This paper shows that Isolation Distributional Kernel (IDK), which is based on a data dependent point kernel, addresses both key issues. We demonstrate IDK's efficacy and efficiency as a new tool for kernel based anomaly detection. Without explicit learning, using IDK alone outperforms existing kernel based anomaly detector OCSVM and other kernel mean embedding methods that rely on Gaussian kernel. We reveal for the first time that an effective kernel based anomaly detector based on kernel mean embedding must employ a characteristic kernel which is data dependent.


- Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks

    Zonghan Wu Shirui Pan Guodong Long Jing Jiang Xiaojun Chang Chengqi Zhang 

Modeling multivariate time series has long been a subject that has attracted researchers from a diverse range of fields including economics, finance, and traffic. A basic assumption behind multivariate time series forecasting is that its variables depend on one another but, upon looking closely, it is fair to say that existing methods fail to fully exploit latent spatial dependencies between pairs of variables. In recent years, meanwhile, graph neural networks (GNNs) have shown high capability in handling relational dependencies. GNNs require well-defined graph structures for information propagation which means they cannot be applied directly for multivariate time series where the dependencies are not known in advance. In this paper, we propose a general graph neural network framework designed specifically for multivariate time series data. Our approach automatically extracts the uni-directed relations among variables through a graph learning module, into which external knowledge like variable attributes can be easily integrated. A novel mix-hop propagation layer and a dilated inception layer are further proposed to capture the spatial and temporal dependencies within the time series. The graph learning, graph convolution, and temporal convolution modules are jointly learned in an end-to-end framework. Experimental results show that our proposed model outperforms the state-of-the-art baseline methods on 3 of 4 benchmark datasets and achieves on-par performance with other approaches on two traffic datasets which provide extra structural information.


- Fast RobustSTL: Efficient and Robust Seasonal-Trend Decomposition for Time Series with Complex Patterns

    Qingsong Wen Zhe Zhang Yan Li Liang Sun 

Many real-world time series data exhibit complex patterns with trend, seasonality, outlier and noise. Robustly and accurately decomposing these components would greatly facilitate time series tasks including anomaly detection, forecasting and classification. RobustSTL is an effective seasonal-trend decomposition for time series data with complicated patterns. However, it cannot handle multiple seasonal components properly. Also it suffers from its high computational complexity, which limits its usage in practice. In this paper, we extend RobustSTL to handle multiple seasonality. To speed up the computation, we propose a special generalized ADMM algorithm to perform the decomposition efficiently. We rigorously prove that the proposed algorithm converges approximately as standard ADMM while reducing the complexity from O(N2) to O(N log N) for each iteration. We empirically study our proposed algorithm with other state-of-the-art seasonal-trend decomposition methods, including MSTL, STR, TBATS, on both synthetic and real-world datasets with single and multiple seasonality. The experimental results demonstrate the superior performance of our decomposition algorithm in terms of both effectiveness and efficiency.


- USAD: UnSupervised Anomaly Detection on Multivariate Time Series

    Julien Audibert Pietro Michiardi Frédéric Guyard Sébastien Marti Maria A. Zuluaga 

The automatic supervision of IT systems is a current challenge at Orange. Given the size and complexity reached by its IT operations, the number of sensors needed to obtain measurements over time, used to infer normal and abnormal behaviors, has increased dramatically making traditional expert-based supervision methods slow or prone to errors. In this paper, we propose a fast and stable method called UnSupervised Anomaly Detection for multivariate time series (USAD) based on adversely trained autoencoders. Its autoencoder architecture makes it capable of learning in an unsupervised way. The use of adversarial training and its architecture allows it to isolate anomalies while providing fast training. We study the properties of our methods through experiments on five public datasets, thus demonstrating its robustness, training speed and high anomaly detection performance. Through a feasibility study using Orange's proprietary data we have been able to validate Orange's requirements on scalability, stability, robustness, training speed and high performance.


- Robust Deep Learning Methods for Anomaly Detection

    Raghavendra Chalapathy Nguyen Lu Dang Khoa Sanjay Chawla 

Anomaly detection is an important problem that has been well-studied within diverse research areas and application domains. A robust anomaly detection system identifies rare events and patterns in the absence of labelled data. The identified patterns provide crucial insights about both the fidelity of the data and deviations in the underlying data-generating process. For example a surveillance system designed to monitor the emergence of new epidemics will use a robust anomaly detection methods to separate spurious associations from genuine indicators of an epidemic with minimal lag time.

The key concept in anomaly detection is the notion of "robustness'', i.e., designing models and representations which are less-sensitive to small changes in the underlying data distribution. The canonical example is that the median is more robust than the mean as an estimator. The tutorial will primarily help researchers and developers design deep learning architectures and loss functions where the learnt representation behave more like the "median'' rather than the "mean.'' The tutorial will revisit well known unsupervised learning techniques in deep learning including autoencoders and generative adversarial networks (GANs) from the perspective of anomaly detection. This in turn will give the audience a more grounded perspective on unsupervised deep learning methods. All the methods will be introduced in a hands-on manner to demonstrate how high-level ideas and concepts get translated to practical real code.



- Deep Learning for Anomaly Detection

    Ruoying Wang Kexin Nie Yen-Jung Chang Xinwei Gong Tie Wang Yang Yang Bo Long 

Anomaly detection has been widely studied and used in diverse applications. Building an effective anomaly detection system requires researchers and developers to learn complex structure from noisy data, identify dynamic anomaly patterns, and detect anomalies with limited labels. Recent advancements in deep learning techniques have greatly improved anomaly detection performance, in comparison with classical approaches, and have extended anomaly detection to a wide variety of applications. This tutorial will help the audience gain a comprehensive understanding of deep learning based anomaly detection techniques in various application domains. First, we give an overview of the anomaly detection problem, introducing the approaches taken before the deep model era and listing out the challenges they faced. Then we survey the state-of-the-art deep learning models that range from building block neural network structures such as MLP, CNN, and LSTM, to more complex structures such as autoencoder, generative models (VAE, GAN, Flow-based models), to deep one-class detection models, etc. In addition, we illustrate how techniques such as transfer learning and reinforcement learning can help amend the label sparsity issue in anomaly detection problems and how to collect and make the best use of user labels in practice. Second to last, we discuss real world use cases coming from and outside LinkedIn. The tutorial concludes with a discussion of future trends.
 
 
### How to build AI platform

https://medium.com/@louisdorard/an-overview-of-ml-development-platforms-df953060b9a9 这里有几个分类有点意思 - Done

- Semi-specialized platforms (e.g. for text or image inputs)

- High-level platforms as a service (mostly for tabular data)

- Self-hosted studios

- Cloud Machine Learning IDEs

https://medium.com/@louisdorard/architecture-of-a-real-world-machine-learning-system-795254bec646 --系列二 - Done




