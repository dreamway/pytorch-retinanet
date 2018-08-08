import os
import shutil

root_dir = '/home/jingwenlai/data/voc/VOCdevkit/VOC2012/JPEGImages'
def query_file(filename):
    filename = os.path.join(root_dir,filename)    
    if os.path.exists(filename):
        return True
    return False

def load_and_query(filename):
    val_file = open(filename,'r')
    for line in val_file.readlines():
        words = line.split(' ')
        name = words[0]

        if query_file(name)==False:
            #print("warning: file not exists", name)
            pass
        else:
            print("file exists",name)
        
import sys
if __name__ == '__main__':
    load_and_query(sys.argv[1])
