from utils.TaskSampler import UniformTaskSampler
from utils.tools import mask_tokens, makeTrainExamples

from utils.printHelper import *
from utils.Logger import logger
from utils.commonVar import *
import logging
import torch
import numpy as np
import copy
import os
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import TensorDataset
import time

import pdb

##
# @brief  base class of evaluator
class StatisticsCollectorBase():
    def __init__(self):
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)


##
# @brief StatisticsCollector used to collect feature space statistics. 
class StatisticsCollector(StatisticsCollectorBase):
    def __init__(self, collectParam):
        super(StatisticsCollector, self).__init__()
        self.beforeBatchNorm = collectParam['beforeBatchNorm']
        logger.info(f'In stat collector, beforeBatchNorm: {self.beforeBatchNorm}.')

    ##
    # @brief 
    #
    # @param c a unit vector
    # @param features
    #
    # @return 
    def partitionFunc(self, c, features):
        summed = 0 
        for feature in features:
            summed += np.exp(np.dot(c, feature))
        return summed


    ##
    # @brief collect parition function value. It is a statistic to measure the isotropy of embedding space.
    #
    # @param model
    #
    # @return 
    def collectIsotropyMeature(self, model, tokenizer, dataset, centralize=False):
        # 1. get sentence of word embeddings in the test dataset
        # 2. collect eigenvectors of V: feature matrix
        # 3. for each eigenvector c, calculate Z(c)
        # 4. the parition value = min Z(c) / max Z(c)
        model.eval()

        startTime = time.time()
        with torch.no_grad():
            # 1. get sentence of word embeddings in the test dataset
            labTensorData = makeTrainExamples(dataset.getTokList(), tokenizer, dataset.getLabID())
            batchSize = 64
            torchDataLoader = torch.utils.data.DataLoader(labTensorData, batch_size = batchSize, shuffle=False)
            embeddingsList = []
            for dataBatch in tqdm(torchDataLoader):
                Y, ids, types, masks = dataBatch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}

                # forward
                embeddings = model.forwardEmbedding(X, beforeBatchNorm=self.beforeBatchNorm)

                embeddingsList.append(embeddings)

            # 2.0 centralize
            isTensor = True
            if isTensor:
                embeddingsNp = torch.cat(embeddingsList).cpu().numpy()
            else:
                embeddingsNp = np.concatenate(embeddingsList)
            logger.info("Embedding vector shape: %d, %d", embeddingsNp.shape[0], embeddingsNp.shape[1])
            if centralize:
                embeddingsNp = embeddingsNp - embeddingsNp.mean(0)

            # 2. collect eigenvectors of V: feature matrix
            square = embeddingsNp.transpose() @ embeddingsNp
            eigenValues, eigenV = np.linalg.eig(square)

            # 3. for each eigenvector c, calculate Z(c)
            funcValues = []
            for v in eigenV:
                funcValue = self.partitionFunc(v, embeddingsNp)
                funcValues.append(funcValue)

            # 4. the measure = min Z(c) / max Z(c)
            measure = min(funcValues)/max(funcValues)
        duration = self.round(time.time() - startTime)

        # acc, pre, rec, F1
        return measure

