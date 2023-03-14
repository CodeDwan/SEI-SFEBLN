# SEI-SFEBLN
This is the open source code for paper: 

**GPU-Free Specific Emitter Identification Using Signal Feature Embedded Broad Learning**

Abstract—Emerging wireless networks may suffer severe security threats due to the ubiquitous access of massive wireless devices. Specific emitter identification (SEI) is considered as one of the important techniques to protect wireless networks, which aims to identifying legal or illegal devices through the radio frequency (RF) fingerprints contained in RF signals. Existing SEI methods are implemented with either traditional machine learning or deep learning. The former relies on manual feature extraction which is usually inefficient, while the latter relies on the powerful GPU computing power but with limited applications and high cost. To solve these problems, in this paper, we propose a GPU-free SEI method using a signal feature embedded broad learning network (SFEBLN), for efficient emitter identification based on a single-layer forward propagation network on the CPU platform. With this method, the original RF data is first pre- processed through external signal processing nodes, and then processed to generate mapped feature nodes and enhancement nodes by nonlinear transformation. Next, we design the internal signal processing nodes to extract effective features from the processed RF signals. The final input layer consists of mapped feature nodes, enhancement nodes and internal signal processing nodes. Then, the network weight parameters are obtained by solving the pseudo inverse problem. Experiments are conducted over the CPU platform and the results show that our proposed SEI method using SFEBLN achieves a superior identification performance and robustness under various scenarios.

**NOTE: The dataset can be download as following link.**
链接:https://pan.baidu.com/s/1R4YubSDIc6jVgGU7IqBM0A?pwd=a343

** If you agree with our work, please consider our paper as a reference. **
\bibitem{ZhangIoT2023a}
Y. Zhang, Y. Peng, J. Sun, G. Gui, Y. Lin, and S. Mao,  ``GPU-Free Specific Emitter Identification Using Signal Feature Embedded Broad Learning,'' \emph{IEEE Internet Things J.}, early access, doi: 10.1109/JIOT.2023.3257479
