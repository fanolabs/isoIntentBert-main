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
# dataDir=hwu64_publishedPaper
# targetDomain='play,lists,recommendation,iot,general,transport,weather,social,email,music,qa,takeaway,audio,news,datetime,calendar,cooking,alarm'
beforeBatchNorm=--beforeBatchNorm


# model initialization
seedModelName=(1 2 3 4 5)

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


for seedName in ${seedModelName[@]}
do
    echo "Start Experiment ..."
    LMName=seed${seedName}_CLWeight1.7_temp0.05_corRegW0.04
    logFolder=./log/
    mkdir -p ${logFolder}
    logFile=${logFolder}/reportIso_${dataDir}_LM${LMName}.log
    if $debug
    then
        logFlie=${logFolder}/logDebug.log
    fi


    echo $LMName 

    export CUDA_VISIBLE_DEVICES=${cudaID}
    python  reportIsotropy.py \
        --seed ${seed} \
        --targetDomain ${targetDomain} \
        --dataDir ${dataDir} \
        --LMName ${LMName} \
        ${beforeBatchNorm}  \
        | tee "${logFile}"
    done
echo "Experiment finished."
