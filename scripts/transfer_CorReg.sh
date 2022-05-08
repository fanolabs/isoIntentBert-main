#!/usr/bin/env bash

echo usage: 
echo scriptName.sh : run in normal mode
echo scriptName.sh debug : run in debug mode

# hardware
cudaID=$2

# debug mode
if [[ $# != 0 ]] && [[ $1 == "debug" ]]
then
    debug=true
else
    debug=false
fi

seedList=(1 2 3 4 5)
seedList=(1)

# dataset
dataDir='OOS,HINT3'

# sourceDomain="utility,home,meta,small_talk,travel,kitchen_dining"
# valDomain="auto_commute,work"
sourceDomain="utility,auto_commute,work,home,meta,small_talk"
valDomain="travel,kitchen_dining"
# valDomain="curekart,powerplay11,sofmattress"
# valDomain="BANKING"

# below is only for evaluation
# targetDomain="BANKING"
targetDomain="curekart,powerplay11,sofmattress"

# setting
epochs=3
taskNum=200
shot=2

# training
simTemp=0.05
lossContrastiveWeight=0.0
lossCorRegWeight=0.04

weightDecay=0.001
validation=--validation
learningRate=2e-5
batchMonitor=50
patience=20

# model setting
# common
LMName=bert-base-uncased
tokenizer=bert-base-uncased
beforeBatchNorm=--beforeBatchNorm

saveModel=--saveModel

# modify arguments if it's debug mode
RED='\033[0;31m'
GRN='\033[0;32m'
NC='\033[0m' # No Color
if $debug
then
    echo -e "Run in ${RED} debug ${NC} mode."
    epochs=2
    disableCuda=--disableCuda
    taskNum=1
else
    echo -e "Run in ${GRN} normal ${NC} mode."
fi

echo "Start Experiment ..."
for seed in ${seedList[@]}
do
    logFolder=./log/
    mkdir -p ${logFolder}
    logFile=${logFolder}/transfer_${sourceDomain}_${way}way_${shot}_seed${seed}_CLWeight${lossContrastiveWeight}_temp${simTemp}_corRegW${lossCorRegWeight}.log
    if $debug
    then
        logFlie=${logFolder}/logDebug.log
    fi

    saveName=seed${seed}_CLWeight${lossContrastiveWeight}_temp${simTemp}_corRegW${lossCorRegWeight}

    export CUDA_VISIBLE_DEVICES=${cudaID}
    python transfer.py \
        --seed ${seed} \
        --epochs ${epochs} \
        --valDomain  ${valDomain}  \
        --sourceDomain ${sourceDomain} \
        --targetDomain ${targetDomain} \
        --dataDir ${dataDir} \
        --shot ${shot}  \
        ${saveModel} \
        ${validation} \
        --weightDecay  ${weightDecay} \
        --learningRate  ${learningRate} \
        --batchMonitor  ${batchMonitor} \
        --LMName ${LMName} \
        --tokenizer  ${tokenizer}  \
        --saveName ${saveName} \
        --taskNum ${taskNum} \
        ${beforeBatchNorm}   \
        --patience  ${patience} \
        --simTemp  ${simTemp}  \
        --lossCorRegWeight  ${lossCorRegWeight}  \
        --lossContrastiveWeight  ${lossContrastiveWeight}  \
        ${disableCuda}   \
        | tee "${logFile}"
    done
echo "Experiment finished."
