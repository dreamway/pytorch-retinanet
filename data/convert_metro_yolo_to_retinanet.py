import shutil
import os
import glob 
yolo_anno_dir = './metro/ImageSets/'
origin_jpegimgs_dir = './metro/JPEGImages/'
target_retina_dir = './retina-metro/'
import cv2

#width = 1920
#height = 1080
width = height = 416

target_train_file = './metro_trainval.txt'

def convert_yolo_to_VOC(yolo_fmt):
    ## Convert from ((x,y)[center], x, y) to (xmin,xmax,ymin,ymax)
    dw = 1.0/width
    dh = 1.0/height
    xc = yolo_fmt[0]
    yc = yolo_fmt[1]
    w = yolo_fmt[2]
    h = yolo_fmt[3]
    x = xc/dw
    y = yc/dh
    w = w/dw
    h = h/dh
    xmin = int(min(x-w/2, x+w/2))
    xmax = int(max(x-w/2, x+w/2))
    ymin = int(min(y-h/2, y+h/2))
    ymax = int(max(y-h/2, y+h/2))

    return (xmin, xmax, ymin, ymax)


def read_yolo_format(yolo_anno_dir):    
    """
    read yolo anno files & convert to retinanet training format, copy images to destination dir.
    """
    outfile = open(target_train_file, 'w')
    anno_files = glob.glob(yolo_anno_dir+"cctv*.txt")
    for anno_file in anno_files:
        bidx = anno_file.rfind('/')
        eidx = anno_file.rfind('.')
        name_id = anno_file[bidx+1:eidx].strip()
        anno_file = open(anno_file, 'r')      

        img_fn = yolo_anno_dir+name_id+".jpg"        
        target_fn = target_retina_dir+name_id+'.jpg'        
          
        outfile.write(name_id+".jpg ")
        for line in anno_file.readlines():
            cls, x, y, w, h = line.split(' ')
            (xmin, xmax, ymin, ymax) = convert_yolo_to_VOC((float(x),float(y),float(w),float(h)))
            print(xmin, xmax, ymin, ymax)    
            anno = str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax)+" "+str(cls)+" "
            outfile.write(anno)
        outfile.write('\n')
        shutil.copy(img_fn, target_fn)

            
if __name__ == '__main__':
    read_yolo_format(yolo_anno_dir)

