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

seed=1

# dataset
dataDir=bank77
targetDomain="BANKING"
# dataDir=HINT3
# targetDomain='curekart,powerplay11,sofmattress'
# dataDir=OOS
# targetDomain='travel,kitchen_dining'
# dataDir=hwu64_publishedPaper
# targetDomain='play,lists,recommendation,iot,general,transport,weather,social,email,music,qa,takeaway,audio,news,datetime,calendar,cooking,alarm'
beforeBatchNorm=--beforeBatchNorm

# setting
way=5
shot=2

# model initialization
seedModelName=(1 2 3 4 5)
CLWeight=0.0
temp=0.05
lossCorRegWeight=0.0

tokenizer=bert-base-uncased

# modify arguments if it's debug mode
RED='\033[0;31m'
GRN='\033[0;32m'
NC='\033[0m' # No Color
if $debug
then
    echo -e "Run in ${RED} debug ${NC} mode."
    epochs=1
else
    echo -e "Run in ${GRN} normal ${NC} mode."
fi

echo "Start Experiment ..."
for seedName in ${seedModelName[@]}
do
    LMName=seed${seedName}_CLWeight${CLWeight}_temp${temp}_corRegW${lossCorRegWeight}
    logFolder=./log/
    mkdir -p ${logFolder}
    logFile=${logFolder}/eval_targetDataset${dataDir}_${way}way_${shot}shot_${LMName}.log
    if $debug
    then
        logFlie=${logFolder}/logDebug.log
    fi

    export CUDA_VISIBLE_DEVICES=${cudaID}
    python eval.py \
        --seed ${seed} \
        --targetDomain ${targetDomain} \
        --tokenizer  ${tokenizer}   \
        --dataDir ${dataDir} \
        --shot ${shot}  \
        --LMName ${LMName} \
        ${beforeBatchNorm}  \
        | tee "${logFile}"
    done
echo "Experiment finished."
