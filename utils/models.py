#coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from transformers import AutoModelForMaskedLM
from utils.commonVar import *
from utils.Logger import logger

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class IntentBERT(nn.Module):
    def __init__(self, config):
        super(IntentBERT, self).__init__()
        self.device = config['device']
        self.LMName = config['LMName']
        self.clsNum = config['clsNumber']
        self.featureDim = 768

        self.linearClsfier = nn.Linear(self.featureDim, self.clsNum)
        self.dropout = nn.Dropout(0.1) # follow the default in bert model
        # self.word_embedding = nn.DataParallel(self.word_embedding)
        # batch norm
        self.BN = nn.BatchNorm1d(num_features=self.featureDim)

        # load from Huggingface or from disk
        try:
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(self.LMName)
        except:
            modelPath = os.path.join(SAVE_PATH, self.LMName)
            logger.info("Loading model from %s"%(modelPath))
            self.word_embedding = AutoModelForMaskedLM.from_pretrained(modelPath)
            BNPath = modelPath + '/' + 'BN.pt'
            self.BN.load_state_dict(torch.load(BNPath))

        self.word_embedding.to(self.device)
        self.linearClsfier.to(self.device)
        self.BN = self.BN.to(self.device)

        logger.info("Contrastive-learning-based reg: temperature %f."%(config['simTemp']))
        self.sim = Similarity(config['simTemp'])


    def loss_contrastive():
        contrastiveLoss = nn.CrossEntropyLoss()
        output = contrastiveLoss()

    def loss_covariance(self, embeddings):
        # covariance
        meanVector = embeddings.mean(dim=0)
        centereVectors = embeddings - meanVector

        # estimate covariance matrix
        featureDim = meanVector.shape[0]
        dataCount = embeddings.shape[0]
        covMatrix = ((centereVectors.t())@centereVectors)/(dataCount-1) 

        # normalize covariance matrix
        stdVector = torch.std(embeddings, dim=0)
        sigmaSigmaMatrix = (stdVector.unsqueeze(1))@(stdVector.unsqueeze(0))
        normalizedConvMatrix = covMatrix/sigmaSigmaMatrix

        deltaMatrix = normalizedConvMatrix - torch.eye(featureDim).to(self.device)

        covLoss = torch.norm(deltaMatrix)   # Frobenius norm
        
        return covLoss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output

    def loss_mse(self, logits, Y):
        loss = nn.MSELoss()
        output = loss(torch.sigmoid(logits).squeeze(), Y)
        return output

    def loss_kl(self, logits, label):
        # KL-div loss
        probs = F.log_softmax(logits, dim=1)
        # label_probs = F.log_softmax(label, dim=1)
        loss = F.kl_div(probs, label, reduction='batchmean')
        return loss

    def getUttEmbeddings(self, X, beforeBatchNorm):
        # BERT forward
        outputs = self.word_embedding(**X, output_hidden_states=True)

        # extract [CLS] for utterance representation
        if beforeBatchNorm:
            CLSEmbedding = outputs.hidden_states[-1][:,0]
        else:
            CLSEmbedding = outputs.hidden_states[-1][:,0]
            CLSEmbedding = self.BN(CLSEmbedding)
            CLSEmbedding = self.dropout(CLSEmbedding)

        return CLSEmbedding


    def forwardEmbedding(self, X, beforeBatchNorm=False):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X, beforeBatchNorm = beforeBatchNorm)

        return CLSEmbedding
    
    def forward(self, X, returnEmbedding=False, beforeBatchNorm=False):
        # get utterances embeddings
        CLSEmbedding = self.getUttEmbeddings(X, beforeBatchNorm = beforeBatchNorm)

        # linear classifier
        logits = self.linearClsfier(CLSEmbedding)

        if returnEmbedding:
            return logits, CLSEmbedding
        else:
            return logits

    def fewShotPredict(self, supportX, supportY, queryX, clsFierName,  beforeBatchNorm=False):
        # calculate word embedding
        supportEmbedding = self.getUttEmbeddings(supportX, beforeBatchNorm)
        queryEmbedding   = self.getUttEmbeddings(queryX, beforeBatchNorm)

        # select clsfier
        support_features = supportEmbedding.cpu()
        query_features = queryEmbedding.cpu()
        clf = None
        if clsFierName == CLSFIER_LINEAR_REGRESSION:
            clf = LogisticRegression(penalty='l2',
                                     random_state=0,
                                     C=1.0,
                                     solver='lbfgs',
                                     max_iter=1000,
                                     multi_class='multinomial')
            # fit and predict
            clf.fit(support_features, supportY)
        elif clsFierName == CLSFIER_SVM:
            clf = make_pipeline(StandardScaler(), 
                                SVC(gamma='auto',C=1,
                                kernel='linear',
                                decision_function_shape='ovr'))
            # fit and predict
            clf.fit(support_features, supportY)
        else:
            raise NotImplementedError("Not supported clasfier name %s", clsFierName)
        
        query_pred = clf.predict(query_features)

        return query_pred
    
    def reinit_clsfier(self):
        self.linearClsfier.weight.data.normal_(mean=0.0, std=0.02)
        self.linearClsfier.bias.data.zero_()
    
    def set_dropout_layer(self, dropout_rate):
        self.dropout = nn.Dropout(dropout_rate)
    
    def set_linear_layer(self, clsNum):
        self.linearClsfier = nn.Linear(768, clsNum)

    def NN(self, support, support_ys, query):
        """nearest classifier"""
        support = np.expand_dims(support.transpose(), 0)
        query = np.expand_dims(query, 2)

        diff = np.multiply(query - support, query - support)
        distance = diff.sum(1)
        min_idx = np.argmin(distance, axis=1)
        pred = [support_ys[idx] for idx in min_idx]
        return pred

    def CosineClsfier(self, support, support_ys, query):
        """Cosine classifier"""
        support_norm = np.linalg.norm(support, axis=1, keepdims=True)
        support = support / support_norm
        query_norm = np.linalg.norm(query, axis=1, keepdims=True)
        query = query / query_norm

        cosine_distance = query @ support.transpose()
        max_idx = np.argmax(cosine_distance, axis=1)
        pred = [support_ys[idx] for idx in max_idx]
        return pred

    def save(self, path):
        # pre-trained LM
        self.word_embedding.save_pretrained(path)

        # BN
        BNPath = path + '/' + 'BN.pt'
        torch.save(self.BN.state_dict(), BNPath)

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

