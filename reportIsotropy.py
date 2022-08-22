# This file assembles three popular metric learnign baselines, matching network, prototype network and relation network.
# This file is coded based on train_matchingNet.py.
# coding=utf-8
import torch
import argparse
import time
from transformers import AutoTokenizer

from utils.StatisticsCollector import StatisticsCollector
from utils.models import IntentBERT
from utils.IntentDataset import IntentDataset
from utils.Evaluator import FewShotEvaluator
from utils.commonVar import *
from utils.printHelper import *
from utils.tools import *
from utils.Logger import logger

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def parseArgument():
    # ==== parse argument ====
    parser = argparse.ArgumentParser(description='Evaluate few-shot performance')

    # ==== model ====
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--mode', default='multi-class',
                        help='Choose from multi-class|entailment-utt|entailment-lab')
    parser.add_argument('--tokenizer', default='bert-base-uncased',
                        help="Name of tokenizer")
    parser.add_argument('--LMName', default='bert-base-uncased',
                        help='Name for models and path to saved model')
    parser.add_argument('--multi_label', action="store_true")
    parser.add_argument('--lossCovWeight', type=float, default=0.5)
    parser.add_argument('--covThreshold', type=float, default=150.0)
    parser.add_argument('--numSynthesis', default=1, type=int)
    parser.add_argument('--simTemp', default=0.05, type=float, help="The temperature of similarity during contrastive learning.")

    # ==== dataset ====
    parser.add_argument('--dataDir',
                        help="Dataset names included in this experiment and separated by comma. "
                        "For example:'OOS,bank77,hwu64'")
    parser.add_argument('--targetDomain',
                        help='Target domain names and separated by comma')
    

    # ==== arguments ====
    parser.add_argument('--beforeBatchNorm', help='Use the features before batch norm to infer or not', action="store_true")
    parser.add_argument('--disableCuda', action="store_true")
    
    # ==== other things ====
    parser.add_argument('--loggingLevel', default='INFO',
                        help="python logging level")

    args = parser.parse_args()

    return args

def main():
    # ======= process arguments ======
    args = parseArgument()
    print(args)

    if args.multi_label:
        args.clsFierName = "MultiLabel"

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
    dataset = IntentDataset(multi_label=args.multi_label)
    dataset.loadDataset(splitName(args.dataDir))
    dataset.tokenize(tok)
    logger.info("----- Testing Data -----")
    testData = dataset.splitDomain(splitName(args.targetDomain), multi_label=args.multi_label)

    # ======= prepare model ======
    # initialize model
    modelConfig = {}
    modelConfig['device'] = torch.device('cuda:0' if not args.disableCuda else 'cpu')
    modelConfig['clsNumber'] = testData.getLabNum()
    modelConfig['LMName'] = args.LMName
    modelConfig['lossCovWeight'] = args.lossCovWeight
    modelConfig['simTemp'] = args.simTemp

    model = IntentBERT(modelConfig)
    logger.info("----- IntentBERT initialized -----")

    # setup evaluator
    collectParam = {'beforeBatchNorm':args.beforeBatchNorm}
    statCollector = StatisticsCollector(collectParam)
    logger.info("Collecting statistics ...")
    # evaluate before finetuning begins
    isotropyMeasure = statCollector.collectIsotropyMeature(model, tok, testData, centralize=True)
    logger.info("Isotropy measure: %f"%(isotropyMeasure))

    # print config
    logger.info(args)
    logger.info(time.asctime())

if __name__ == "__main__":
    main()
    exit(0)
