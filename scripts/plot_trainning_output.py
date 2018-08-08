import matplotlib.pyplot as plt
import numpy as np

def plot_data(data, label):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)    
    ax.set_title(label)
    ax.set_xlabel('epochs')
    ax.plot(data)
    plt.show()


def extract_log(log, keyword1, keyword2, keyword3, keyword4):
    avg_list = []
    cls_list = []
    train_list = []
    loc_list = []

    for line in log.readlines():
        if keyword4 in line:
            eidx = line.find('|')
            bidx = line.find('loc_loss:')
            loc_str = line[bidx + 9:eidx].strip()
            if loc_str == 'nan':
                loc = -1.0
                loc_list.append(loc)
            else:
                try:
                    loc = float(loc_str)
                    loc_list.append(loc)
                except:
                    loc_list.append(-2.0)
        if keyword2 in line:
            bidx = line.find('cls_loss:')
            substr = line[bidx:]
            eidx = substr.find('|') + bidx
            cls_str = line[bidx + 9:eidx].strip(' ').rstrip(' ')
            print(cls_str)
            if 'nan' in cls_str:
                cls = -1.0
                cls_list.append(cls)
            else:
                try:
                    cls = float(cls_str)
                    cls_list.append(cls)
                except:
                    print("ERROR", cls_str)
                    cls_list.append(-2.0)
        if keyword3 in line:
            bidx = line.find('train_loss')
            substr = line[bidx:]
            eidx = substr.find('|') + bidx
            train_str = line[bidx + 11:eidx].strip(' ').rstrip(' ')
            if train_str == 'nan':
                train = -1.0
                train_list.append(train)
            else:
                try:
                    train = float(train_str)
                    train_list.append(train)
                except:
                    train_list.append(-2.0)
        if keyword1 in line:
            bidx = line.find('avg_loss')
            avg_str = line[bidx + 9:]
            avg_str.strip()
            if avg_str == 'nan':
                avg = -1.0
                avg_list.append(avg)
            else:
                try:
                    avg = float(avg_str)
                    avg_list.append(avg)
                except:
                    avg_list.append(-2.0)

    return avg_list, cls_list, train_list, loc_list

def extract_train_test_loss(log):
    train_loss = []
    avg_loss = []
    test_loss = []
    for line in log.readlines():
        if line.startswith('epoch train_loss'):
            bix = line.find('train_loss:')
            eix = line.rfind(',')
            trn_loss = float(line[bix+11:eix].strip())
            train_loss.append(trn_loss)
            bix = line.find('avg_loss:')
            avg = float(line[bix+9:].strip())
            avg_loss.append(avg)
        if 'test_loss:' in line and line.startswith('epoch'):
            bix = line.find('test_loss:')
            loss = float(line[bix+10:].strip())
            test_loss.append(loss)
    return train_loss, avg_loss, test_loss

import sys

def main():    
    log_file = sys.argv[1]
    log = open(log_file,'r')
   
    '''
    avg, cls, train, loc = extract_log(log, 'avg_loss', 'cls_loss', 'train_loss', 'loc_loss')
    plot_data(avg, 'avg_loss')
    plot_data(cls, 'cls_loss')
    plot_data(train, 'train_loss')
    plot_data(loc, 'loc_loss')
    '''
    train_loss, avg_loss, test_loss = extract_train_test_loss(log)
    plot_data(train_loss, 'train_loss')
    plot_data(avg_loss, 'avg_loss')
    plot_data(test_loss, 'test_loss')
    log.close()

if __name__ == '__main__':
    main()

