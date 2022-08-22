# coding=utf-8
import argparse
import time
import subprocess
import os
import numpy as np

def parseArgument():
    # ==== parse argument ====
    parser = argparse.ArgumentParser(description='Evaluate few-shot performance')

    # ==== model ====
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--weight')
    parser.add_argument('--weight2')
    parser.add_argument('--temp')
    parser.add_argument('--logPattern')
    parser.add_argument('--dataPattern')
    parser.add_argument('--shot')

    args = parser.parse_args()

    return args

def main():
    # ======= process arguments ======
    args = parseArgument()
    print(args)

    # count
    #  cmd = 'ls ' +  ' | grep ' + args.logPattern + ' | grep '  + ' seed' + str(args.seed) + " | " + 'grep -i ' + args.dataPattern + ' | '+ 'xargs grep -i ' + args.strPattern + ' | ' + 'wc'
    # cmd = 'ls ' +  ' | grep ' + args.logPattern + ' | grep '  + ' Weight' + str(args.weight) + " | " + 'grep -i ' + args.dataPattern + ' | '+ 'xargs grep -i ' + args.strPattern + ' | ' + 'wc'
    # print("cmd: " + cmd)
    # rt = subprocess.check_output(cmd, shell=True).decode('ascii')
    # lineCount = int(rt.split()[0])
    # print("Data number: %d"%(lineCount))

    mode = 1  # best dev acc
    if args.logPattern == 'report':
        mode = 2
    elif  args.logPattern == 'eval':
        mode = 3
    else:
        print(f"Invalid logPattern {args.logPattern}.")

    if mode == 1:
        strPattern = 'best_'
        # get lines
        cmd = 'ls ' +  ' | grep ' + args.logPattern + ' | grep '  + ' Weight' + str(args.weight) + " | " + 'grep -i ' + args.dataPattern + ' | '+ 'xargs grep -i ' + strPattern + ' | ' + 'awk \'{print $0}\''
        print("cmd: " + cmd)
        rt = subprocess.check_output(cmd, shell=True).decode('ascii').rstrip()
        print("Returned data: ")
        print(rt)

        # get values
        cmd = 'ls ' +  ' | grep ' + args.logPattern + ' | grep '  + ' Weight' + str(args.weight) + " | " + 'grep -i ' + args.dataPattern + ' | '+ 'xargs grep -i ' + strPattern + ' | ' + 'awk \'{print $3}\''
        print("cmd: " + cmd)
        rt = subprocess.check_output(cmd, shell=True).decode('ascii').rstrip()
        print("Returned data: ")
        print(rt)
        valueStrList = rt.split()
        valuesNp = np.asarray([float(valueStr) for valueStr in valueStrList])
        print("Value count: %d"%(valuesNp.shape[0]))
        print("Value mean:  %f"%(valuesNp.mean()))
        print("Value std:   %f"%(valuesNp.std()))
        print(args)
    elif mode == 2:
        # get lines
        cmd = 'ls ' +  ' | grep ' + args.logPattern + ' | grep '  + '_CLWeight' + str(args.weight) + " | " + 'grep -i ' + args.dataPattern + ' | ' + ' grep ' + 'temp' + args.temp +  ' | ' +  ' grep ' + 'corRegW' + args.weight2 + '.log' + ' | ' +  'xargs grep -i ' + 'measure' + ' | ' + 'awk \'{print $0}\''
        print("cmd: " + cmd)
        rt = subprocess.check_output(cmd, shell=True).decode('ascii').rstrip()
        print("Returned data: ")
        print(rt)

        # get values
        cmd = 'ls ' +  ' | grep ' + args.logPattern + ' | grep '  + ' _CLWeight' + str(args.weight) + " | " + 'grep -i ' + args.dataPattern +  ' | ' + ' grep ' + 'temp' + args.temp + ' | '  + ' grep ' + 'corRegW' + args.weight2 + '.log' + ' | ' + 'xargs grep -i ' + 'measure' + ' | ' + 'awk \'{print $4}\''
        print("cmd: " + cmd)
        rt = subprocess.check_output(cmd, shell=True).decode('ascii').rstrip()
        print("Returned data: ")
        print(rt)
        valueStrList = rt.split()
        valuesNp = np.asarray([float(valueStr) for valueStr in valueStrList])
        print("Value count: %d"%(valuesNp.shape[0]))
        print("Value mean:  %f"%(valuesNp.mean()))
        print("Value std:   %f"%(valuesNp.std()))
        print("Value mean +- std: %f +- %f "%(valuesNp.mean(), valuesNp.std()))
        print(args)
    elif mode == 3:
        # get lines
        cmd = 'ls ' +  ' | grep ' + args.logPattern +  " | grep " + args.shot + "shot" + " | " + 'grep -i ' + args.dataPattern + ' | ' + 'grep ' + 'CLWeight' + args.weight + '_' + ' | ' + ' grep temp' + args.temp + ' | ' + ' grep corRegW' + args.weight2 + '.log' + ' | '   + ' xargs grep -i ' + 'acc' + ' | ' + 'awk \'{print $0}\''
        print("cmd: " + cmd)
        rt = subprocess.check_output(cmd, shell=True).decode('ascii').rstrip()
        print("Returned data: ")
        print(rt)

        # get values
        cmd = 'ls ' +  ' | grep ' + args.logPattern + " | grep " + args.shot + "shot" + " | "  + 'grep -i ' + args.dataPattern + ' | '+ 'grep ' + 'CLWeight' + args.weight + '_' +  ' | ' + ' grep temp' + args.temp + ' | '  + ' grep corRegW' + args.weight2 + '.log' + ' | ' + ' xargs grep -i ' + '\'acc   :\'' + ' | ' + 'awk \'{print $4}\''
        print("cmd: " + cmd)
        rt = subprocess.check_output(cmd, shell=True).decode('ascii').rstrip()
        print("Returned data: ")
        print(rt)
        valueStrList = rt.split()
        valuesNp = np.asarray([float(valueStr) for valueStr in valueStrList])
        print("Value count: %d"%(valuesNp.shape[0]))
        print("Value mean:  %f"%(valuesNp.mean()))
        print("Value std:   %f"%(valuesNp.std()))
        print("Value mean +- std: %f +- %f "%(valuesNp.mean(), valuesNp.std()))
        print(args)

if __name__ == "__main__":
    main()
    exit(0)
