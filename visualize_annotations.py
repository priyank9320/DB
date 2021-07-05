import cv2
import os
from numpy import genfromtxt
import argparse
from experiment import Structure
import numpy as np


# for visualizing the bounding boxes from annotations

def main():
    parser = argparse.ArgumentParser(description='Visualize Annotations')
    parser.add_argument('--img', type=str, help='path of the image file')
    parser.add_argument('--anno', type=str, help='path of the annotation file')


    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    path='./visualize_annotations_output'
    try: 
        os.mkdir(path) 
        print('folder created: ',path)
    except Exception as e: 
        #print('exception raised if already existing folder',e) 
        pass

    input_boxes = genfromtxt(args['anno'], delimiter=',', dtype=int, usecols=(0,1,2,3,4,5,6,7))
    output=[[input_boxes,0],0]
    vis_image = demo_visualize(image_path=args['img'], output=output)
    cv2.imwrite(os.path.join(path, args['img'].split('/')[-1].split('.')[0]+'.jpg'), vis_image)
    print('image generated and stored in visualize_annotations_output folder')



def demo_visualize(image_path, output):
    boxes, _ = output
    boxes = boxes[0]
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    original_shape = original_image.shape
    pred_canvas = original_image.copy().astype(np.uint8)
    pred_canvas = cv2.resize(pred_canvas, (original_shape[1], original_shape[0]))

    for box in boxes:
        #print('box value ',box)
        box = np.array(box).astype(np.int32).reshape(-1, 2)
        cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)

    return pred_canvas


    
if __name__ == '__main__':
    main()
