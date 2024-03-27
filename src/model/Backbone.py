import numpy as np
import torch
import torch.nn as nn
from .resnet.resnet import resnet101
from .GraphNeuralNetwork import GatedGNN
from .SemanticDecoupling import SemanticDecoupling
from .Element_Wise_Layer import Element_Wise_Layer

class Backbone(nn.Module):

    def __init__(self, adjacencyMatrix, wordFeatures, topK,
                 imageFeatureDim=1024, intermediaDim=512, outputDim=1024,
                 classNum=80, wordFeatureDim=300, timeStep=3):

        super(Backbone, self).__init__()

        self.backbone = resnet101()

        if imageFeatureDim != 2048:
            self.changeChannel = nn.Sequential(nn.Conv2d(2048, imageFeatureDim, kernel_size=1, stride=1, bias=False),
                                               nn.BatchNorm2d(imageFeatureDim),)

        self.classNum = classNum
        self.topK = topK

        # self.poolingType = poolingType
        self.avgPooling = nn.AdaptiveAvgPool2d(1)
        self.maxPooling = nn.AdaptiveMaxPool2d(1)

        self.timeStep = timeStep
        self.outputDim = outputDim
        self.intermediaDim = intermediaDim
        self.wordFeatureDim = wordFeatureDim
        self.imageFeatureDim = imageFeatureDim
        
        self.wordFeatures = self.load_features(wordFeatures)
        self.inMatrix, self.outMatrix = self.load_matrix(adjacencyMatrix)

        self.SemanticDecoupling = SemanticDecoupling(classNum, imageFeatureDim, wordFeatureDim, intermediaDim=intermediaDim) #, poolingType=poolingType) 
        self.GraphNeuralNetwork = GatedGNN(imageFeatureDim, timeStep, self.inMatrix, self.outMatrix) 

        self.fc = nn.Linear(2 * imageFeatureDim, outputDim)
        self.classifiers = Element_Wise_Layer(classNum, outputDim)   


    def forward(self, input, target=None):

        batchSize = input.size(0)

        featureMap = self.backbone(input) # (batchSize, channel, imgSize, imgSize)
        
        if featureMap.size(1) != self.imageFeatureDim:
            featureMap = self.changeChannel(featureMap) # (batchSize, imgFeatureDim, imgSize, imgSize)

        semanticFeature, featuremapWithCoef, coefficient = self.SemanticDecoupling(featureMap, self.wordFeatures)     # (batchSize, classNum, outputDim)
        
        # Predict Category
        feature = self.GraphNeuralNetwork(semanticFeature) # if self.useGatedGNN else semanticFeature
        output = torch.tanh(self.fc(torch.cat((feature.view(batchSize * self.classNum, -1),
                                               semanticFeature.view(-1, self.imageFeatureDim)),1)))

        output = output.contiguous().view(batchSize, self.classNum, self.outputDim)
        result = self.classifiers(output) # (batchSize, classNum)

        if target is None:
            return result


    def load_features(self, wordFeatures):
        return nn.Parameter(torch.from_numpy(wordFeatures.astype(np.float32)), requires_grad=False)

    def load_matrix(self, mat):
        _in_matrix, _out_matrix = mat.astype(np.float32), mat.T.astype(np.float32)
        _in_matrix, _out_matrix = nn.Parameter(torch.from_numpy(_in_matrix), requires_grad=False), nn.Parameter(torch.from_numpy(_out_matrix), requires_grad=False)
        return _in_matrix, _out_matrix

# =============================================================================
# Help Functions
# =============================================================================