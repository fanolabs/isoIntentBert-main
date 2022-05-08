# This file assembles three popular metric learnign baselines, matching network, prototype network and relation network.
# This file is coded based on train_matchingNet.py.
# coding=utf-8
import os
import torch
import torch.optim as optim
import argparse
import time
import copy
from transformers import AutoTokenizer
import random

from utils.models import IntentBERT
from utils.IntentDataset import IntentDataset
from utils.Trainer import TransferTrainer
from utils.Evaluator import FewShotEvaluator
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger
import pdb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parseArgument():
    # ==== parse argument ====
    parser = argparse.ArgumentParser(description='Train IntentBERT')

    # ==== model ====
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--tokenizer', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-uncased',
                        help='Name for models and path to saved model')
    parser.add_argument('--beforeBatchNorm', help='Use the features before batch norm to infer or not', action="store_true")
    parser.add_argument('--simTemp', default=0.05, type=float, help="The temperature of similarity during contrastive learning.")
    parser.add_argument('--lossContrastiveWeight', type=float, default=0.1)
    parser.add_argument('--lossCorRegWeight', type=float, default=0.3)
    
    # ==== dataset ====
    parser.add_argument('--dataDir',
                        help="Dataset names included in this experiment and separated by comma. "
                        "For example:'OOS,bank77,hwu64'")
    parser.add_argument('--sourceDomain',
                        help="Source domain names and separated by comma. "
                        "For example:'travel,banking,home'")
    parser.add_argument('--valDomain',
                        help='Validation domain names and separated by comma')
    parser.add_argument('--targetDomain',
                        help='Target domain names and separated by comma')

    # ==== evaluation task ====
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--query', type=int, default=5)
    parser.add_argument('--clsFierName', default='Linear',
                        help="Classifer name for few-shot evaluation"
                        "Choose from Linear|SVM|NN|Cosine|MultiLabel")

    # ==== optimizer ====
    parser.add_argument('--optimizer', default='Adam',
                        help='Choose from SGD|Adam')
    parser.add_argument('--learningRate', type=float, default=2e-5)
    parser.add_argument('--weightDecay', type=float, default=0)

    # ==== training arguments ====
    parser.add_argument('--disableCuda', action="store_true")
    parser.add_argument('--validation', action="store_true")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--taskNum', type=int, default=500)
    parser.add_argument('--patience', type=int, default=3,
                        help="Early stop when performance does not go better")
    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")
    parser.add_argument('--saveModel', action='store_true',
                        help="Whether to save pretrained model")
    parser.add_argument('--saveName', default='none',
                        help="Specify a unique name to save your model"
                        "If none, then there will be a specific name controlled by how the model is trained")
    parser.add_argument('--batchMonitor', type=int, help='for how many batches to monitor the validation performance', default=50)


    args = parser.parse_args()

    return args

def main():
    # ======= process arguments ======
    args = parseArgument()
    print(args)

    if not args.saveModel:
        logger.info("The model will not be saved after training!")

    # ==== setup logger ====
    if args.loggingLevel == LOGGING_LEVEL_INFO:
        loggingLevel = logging.INFO
    elif args.loggingLevel == LOGGING_LEVEL_DEBUG:
        loggingLevel = logging.DEBUG
    else:
        raise NotImplementedError("Not supported logging level %s", args.loggingLevel)
    logger.setLevel(loggingLevel)

    # ==== set seed ====
    if args.seed >= 0:
        set_seed(args.seed)
        logger.info("The random seed is set %d"%(args.seed))
    else:
        logger.info("The random seed is not set any value.")

    # ======= process data ======
    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # load raw dataset
    logger.info(f"Loading data from {args.dataDir}")
    dataset = IntentDataset()
    dataset.loadDataset(splitName(args.dataDir))
    dataset.tokenize(tok)
    # spit data into training, validation and testing
    logger.info("----- Training Data -----")
    trainData = dataset.splitDomain(splitName(args.sourceDomain))
    logger.info("----- Validation Data -----")
    metaTaskInfo = {'shot':args.shot, 'query':args.query}
    valData = dataset.splitDomain(splitName(args.valDomain), metaTaskInfo = metaTaskInfo)
    logger.info("----- Testing Data -----")
    testData = dataset.splitDomain(splitName(args.targetDomain), metaTaskInfo = metaTaskInfo)

    # ======= prepare model ======
    # initialize model
    modelConfig = {}
    modelConfig['device'] = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    modelConfig['clsNumber'] = trainData.getLabNum()
    modelConfig['LMName'] = args.LMName
    modelConfig['simTemp'] = args.simTemp
    model = IntentBERT(modelConfig)
    logger.info("----- IntentBERT initialized -----")

    # setup validator
    valParam = {"evalTaskNum": args.taskNum, "clsFierName": args.clsFierName, "multi_label": False, 'beforeBatchNorm':args.beforeBatchNorm}
    valTaskParam = {"way":args.way, "shot":args.shot, "query":args.query}
    validator = FewShotEvaluator(valParam, valTaskParam, valData)
    tester = FewShotEvaluator(valParam, valTaskParam, testData)

    # setup trainer
    optimizer = None
    if args.optimizer == OPTER_ADAM:
        optimizer = optim.AdamW(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    elif args.optimizer == OPTER_SGD:
        optimizer = optim.SGD(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    else:
        raise NotImplementedError("Not supported optimizer %s"%(args.optimizer))

    datasetName = args.dataDir.split(',')[0]
    valDatasetName = args.dataDir.split(',')[-1]
    valDatasetName = datasetName
    trainingParam = {"epoch"      : args.epochs, \
                     "batch"      : args.batch_size, \
                     "validation" : args.validation, \
                     "patience"   : args.patience, \
                     "beforeBatchNorm": args.beforeBatchNorm, \
                     "lossContrastiveWeight": args.lossContrastiveWeight, \
                     "lossCorRegWeight": args.lossCorRegWeight, \
                     "batchMonitor":args.batchMonitor, \
                     "wandb": None, \
                     "wandbProj": None, \
                     "wandbRunName": None, \
                     "wandbConfig": None}
    unlabeledData = None
    trainer = TransferTrainer(trainingParam, optimizer, trainData, unlabeledData, validator, tester)

    # train
    trainer.train(model, tok)

    # load best model
    bestModelStateDict = trainer.getBestModelStateDict()
    model.load_state_dict(bestModelStateDict)

    # save model into disk
    if args.saveModel:
        save_path = os.path.join(SAVE_PATH, args.saveName)

        # save
        logger.info("Saving model.pth into folder: %s", save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save(save_path)

    # print config
    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
