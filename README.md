# GAGCN
Code of CVPR 2022 Paper: Spatio-Temporal Gating-Adjacency GCN for Human Motion Prediction

Abstract: Predicting future motion based on historical motion sequence is a fundamental problem in computer vision, and it has wide applications in autonomous driving and robotics. Some recent works have shown that Graph Convo-
lutional Networks(GCN) are instrumental in modeling the relationship between different joints. However, considering the variants and diverse action types in human motion data, the cross-dependency of the spatio-temporal relationships will be difficult to depict due to the decoupled modeling strategy, which may also exacerbate the problem of insufficient generalization. Therefore, we propose the Spatio-Temporal Gating-Adjacency GCN(GAGCN) to learn the complex spatio-temporal dependencies over diverse action types. Specifically, we adopt gating networks to enhance the generalization of GCN via the trainable adaptive adjacency matrix obtained by blending the candidate spatio-temporal adjacency matrices. Moreover, GAGCN addresses the cross-dependency of space and time by balancing the weights of spatio-temporal modeling and fusing the decoupled spatio-temporal features. Extensive experiments on Human 3.6M, AMASS, and 3DPW demonstrate that GAGCN achieves state-of-the-art performance in both short-term and long-term predictions

# Acknowledgements
Part of the code is borrowed from " Space-time-separable graph convolutional network for pose forecasting"
