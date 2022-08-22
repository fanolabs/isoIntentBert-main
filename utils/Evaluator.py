from utils.IntentDataset import IntentDataset
from utils.TaskSampler import MultiLabTaskSampler, UniformTaskSampler
from utils.tools import makeEvalExamples
from utils.printHelper import *
from utils.Logger import logger
from utils.commonVar import *
import logging
import torch
import numpy as np
import pdb
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

##
# @brief  base class of evaluator
class EvaluatorBase():
    def __init__(self):
        self.roundN = 4
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def evaluate(self):
        raise NotImplementedError("train() is not implemented.")

##
# @brief MetaEvaluator used to do meta evaluation. Tasks are sampled and the model is evaluated task by task.
class FewShotEvaluator(EvaluatorBase):
    def __init__(self, evalParam, taskParam, dataset: IntentDataset):
        super(FewShotEvaluator, self).__init__()
        self.way   = taskParam['way']
        self.shot  = taskParam['shot']
        self.query = taskParam['query']

        self.dataset = dataset

        self.clsFierName = evalParam['clsFierName']
        self.evalTaskNum = evalParam['evalTaskNum']
        logger.info("In evaluator classifier %s is used.", self.clsFierName)

        self.beforeBatchNorm = evalParam['beforeBatchNorm']
        logger.info("In evaluator, beforeBatchNorm %s"%(self.beforeBatchNorm))

        self.taskSampler = UniformTaskSampler(self.dataset, self.way, self.shot, self.query)

    def evaluate(self, model, tokenizer, logLevel='DEBUG'):
        model.eval()

        performList = []   # acc, pre, rec, fsc
        with torch.no_grad():
            for task in range(self.evalTaskNum):
                # sample a task
                task = self.taskSampler.sampleOneTask()

                # collect data
                supportX = task[META_TASK_SHOT_TOKEN]
                queryX = task[META_TASK_QUERY_TOKEN]
                supportY = task[META_TASK_SHOT_LOC_LABID]
                queryY = task[META_TASK_QUERY_LOC_LABID]

                # padding
                supportX, supportY, queryX, queryY =\
                    makeEvalExamples(supportX, supportY, queryX, queryY, tokenizer)

                # forward
                queryPrediction = model.fewShotPredict(supportX.to(model.device),
                                                       supportY,
                                                       queryX.to(model.device),
                                                       self.clsFierName,
                                                       beforeBatchNorm=self.beforeBatchNorm)

                
                # calculate acc
                acc = accuracy_score(queryY, queryPrediction)   # acc
                performDetail = precision_recall_fscore_support(queryY, queryPrediction, average='macro', warn_for=tuple())

                performList.append([acc, performDetail[0], performDetail[1], performDetail[2]])
        
        # performance mean and std
        performMean = np.mean(np.stack(performList, 0), 0)
        performStd  = np.std(np.stack(performList, 0), 0)

        if logLevel == 'DEBUG':
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.debug("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.DEBUG)
        else:
            itemList = ["acc", "pre", "rec", "fsc"]
            logger.info("Evaluate statistics: ")
            printMeanStd(performMean, performStd, itemList, debugLevel=logging.INFO)

        # acc, pre, rec, F1
        return performMean[0], performMean[1], performMean[2], performMean[3]
