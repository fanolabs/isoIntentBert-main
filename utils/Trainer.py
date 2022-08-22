from utils.IntentDataset import IntentDataset
from utils.Evaluator import EvaluatorBase
from utils.Logger import logger
from utils.commonVar import *
from utils.tools import mask_tokens, makeTrainExamples
import time
import torch
from torch.utils.data import DataLoader
import numpy as np
import copy
from sklearn.metrics import accuracy_score, r2_score
from torch.utils.tensorboard import SummaryWriter
import wandb
import pdb

##
# @brief  base class of trainer
class TrainerBase():
    def __init__(self, wandb, wandbProj, wandbConfig, wandbRunName):
        self.finished=False
        self.bestModelStateDict = None
        self.roundN = 4
        self.eps = 1e-6

        # wandb 
        self.wandb = wandb
        self.wandbProjName = wandbProj
        self.wandbConfig = wandbConfig
        self.runName = wandbRunName
        pass

    def round(self, floatNum):
        return round(floatNum, self.roundN)

    def train(self):
        raise NotImplementedError("train() is not implemented.")

    def getBestModelStateDict(self):
        return self.bestModelStateDict

class TransferTrainer(TrainerBase):
    def __init__(self,
            trainingParam:dict,
            optimizer,
            dataset:IntentDataset,
            unlabeled:IntentDataset,
            valEvaluator: EvaluatorBase,
            testEvaluator:EvaluatorBase):
        super(TransferTrainer, self).__init__(trainingParam["wandb"], trainingParam["wandbProj"], trainingParam["wandbConfig"], trainingParam["wandbRunName"])
        self.epoch       = trainingParam['epoch']
        self.batch_size  = trainingParam['batch']
        self.validation  = trainingParam['validation']
        self.patience    = trainingParam['patience']
        self.lossContrastiveWeight = trainingParam['lossContrastiveWeight']
        self.lossCorRegWeight = trainingParam['lossCorRegWeight']

        self.dataset       = dataset
        self.unlabeled     = unlabeled
        self.optimizer     = optimizer
        self.valEvaluator  = valEvaluator
        self.testEvaluator = testEvaluator

        self.batchMonitor = trainingParam["batchMonitor"]

        self.beforeBatchNorm = trainingParam['beforeBatchNorm']
        logger.info("In trainer, beforeBatchNorm %s"%(self.beforeBatchNorm))


    ##
    # @brief duplicate input for contrastive learning
    #
    # @param X, a dict {'input_ids':[batch, Len], 'token_type_ids': [batch, Len], 'attention_mask':[batch, Len]}
    #
    # @return  X, a dict {'input_ids':[2*batch, Len], 'token_type_ids': [2*batch, Len], 'attention_mask':[2*batch, Len]}
    def duplicateInput(self, X):
        batchSize = X['input_ids'].shape[0]

        X_duplicate = {}
        X_duplicate['input_ids'] = X['input_ids'].unsqueeze(1).repeat(1,2,1).view(batchSize*2, -1)
        X_duplicate['token_type_ids'] = X['token_type_ids'].unsqueeze(1).repeat(1,2,1).view(batchSize*2, -1)
        X_duplicate['attention_mask'] = X['attention_mask'].unsqueeze(1).repeat(1,2,1).view(batchSize*2, -1)

        return X_duplicate

    ##
    # @brief calculate dropout-based contrastive loss
    #
    # @param model   model
    # @param X       X, a dict {'input_ids':[batch, Len], 'token_type_ids': [batch, Len], 'attention_mask':[batch, Len]}
    # @param beforeBatchNorm: get embeddings before or after batch norm
    #
    # @return  a loss value, tensor
    def calculateDropoutCLLoss(self, model, X, beforeBatchNorm=False):
        # duplicate input
        batch_size = X['input_ids'].shape[0]
        X_dup = self.duplicateInput(X)

        # get raw embeddings
        batchEmbedding = model.forwardEmbedding(X_dup, beforeBatchNorm=beforeBatchNorm)
        batchEmbedding = batchEmbedding.view((batch_size, 2, batchEmbedding.shape[1])) # (bs, num_sent, hidden)

        # Separate representation
        z1, z2 = batchEmbedding[:,0], batchEmbedding[:,1]

        cos_sim = model.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        logits = cos_sim

        labels = torch.arange(logits.size(0)).long().to(model.device)
        lossVal = model.loss_ce(logits, labels)

        return  lossVal

    def train(self, model, tokenizer, mode='multi-class'):
        self.bestModelStateDict = copy.deepcopy(model.state_dict())
        durationOverallTrain = 0.0
        durationOverallVal = 0.0
        valBestAcc = -1
        accumulateStep = 0

        labTensorData = makeTrainExamples(self.dataset.getTokList(), tokenizer, self.dataset.getLabID())
        dataloader = DataLoader(labTensorData, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True)

        earlyStopFlag = False
        for epoch in range(self.epoch):  # an epoch means all sampled tasks are done
            batchTrLossMLMSum = 0.0
            timeEpochStart    = time.time()

            timeMonitorWindowStart = time.time()
            batchNum = len(dataloader)
            for batchID, batch in enumerate(dataloader):
                model.train()
                # task data
                Y, ids, types, masks = batch
                X = {'input_ids':ids.to(model.device),
                        'token_type_ids':types.to(model.device),
                        'attention_mask':masks.to(model.device)}

                # forward
                logits, embeddings = model(X, returnEmbedding=True, beforeBatchNorm=self.beforeBatchNorm)
                # loss
                lossSP = model.loss_ce(logits, Y.to(model.device))
                lossTOT = lossSP

                # covReg
                covLoss = -1
                if self.lossCorRegWeight > self.eps:
                    covLoss = model.loss_covariance(embeddings)
                    lossTOT = lossTOT + self.lossCorRegWeight * covLoss

                # dropout-based contrastive learning
                lossDropoutCLLoss = -1
                if self.lossContrastiveWeight > self.eps:
                    lossDropoutCLLoss = self.calculateDropoutCLLoss(model, X, beforeBatchNorm=self.beforeBatchNorm)
                    lossTOT = lossTOT + self.lossContrastiveWeight * lossDropoutCLLoss

                # backward
                self.optimizer.zero_grad()
                lossTOT.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                self.optimizer.step()

                # calculate train acc
                YTensor = Y.cpu()
                logits = logits.detach().clone()
                if torch.cuda.is_available():
                    logits = logits.cpu()

                logits = logits.numpy()
                predictResult = np.argmax(logits, 1)
                acc = accuracy_score(YTensor, predictResult)

                if (batchID % self.batchMonitor) == 0:
                    # self.batchMonitor number batch training done, collect data
                    model.eval()
                    valAcc, valPre, valRec, valFsc = self.valEvaluator.evaluate(model, tokenizer, logLevel='DEBUG')

                    # statistics
                    monitorWindowDurationTrain = self.round(time.time() - timeMonitorWindowStart)

                    # display current epoch's info
                    logger.info("---- epoch: %d/%d, batch: %d/%d, monitor window time %f ----", epoch, self.epoch, batchID, batchNum, self.round(monitorWindowDurationTrain))
                    logger.info("TrainLoss %f", lossTOT.item())
                    logger.info("valAcc %f, valPre %f, valRec %f , valFsc %f", valAcc, valPre, valRec, valFsc)
                    logger.info("LossCE %f, lossCL %f, covLoss %f.", lossSP, lossDropoutCLLoss, covLoss)

                    # time
                    timeMonitorWindowStart = time.time()
                    durationOverallTrain += monitorWindowDurationTrain

                    # early stop
                    if not self.validation:
                        valAcc = -1
                    if (valAcc >= valBestAcc):   # better validation result
                        print("[INFO] Find a better model. Val acc: %f -> %f"%(valBestAcc, valAcc))
                        valBestAcc = valAcc
                        accumulateStep = 0

                        # cache current model, used for evaluation later
                        self.bestModelStateDict = copy.deepcopy(model.state_dict())
                    else:
                        accumulateStep += 1
                        if accumulateStep > self.patience/2:
                            print('[INFO] accumulateStep: ', accumulateStep)
                            if accumulateStep == self.patience:  # early stop
                                logger.info('Early stop.')
                                logger.debug("Overall training time %f", durationOverallTrain)
                                logger.debug("best_val_acc: %f", valBestAcc)
                                earlyStopFlag = True
                                break

            if earlyStopFlag:
                break

        logger.debug('All %d epochs are finished', self.epoch)
        logger.debug("Overall training time %f", durationOverallTrain)
        logger.info("best_val_acc: %f", valBestAcc)
