import FunctionTool
import argparse
import sys

parser = argparse.ArgumentParser(description='mode prediction')




parser.add_argument('img_dir',action="store",type=str,
                    help='Enter path to sample image EX) flowers/test/1/image_06743.jpg')

parser.add_argument('checkpoint',action="store",type=str,
                    help='Enter path to checkpoint EX) checkpoint.pth')
                    
parser.add_argument('--top_k', dest="top_k", action="store", default=5,
                    help='Slect tok_k to use. Default is 5')
                    
parser.add_argument('--category_names', dest="category_names", action="store", default="cat_to_name.json",
                    help='Enter path to category_names file ./local/cat_to_name.json')

parser.add_argument('--gpu', dest="device", action="store_true", default=False,
                    help='Turn GPU mode on or off, default is off.')        


pa = parser.parse_args()

img_dir        = pa.img_dir
checkpoint     = pa.checkpoint
top_k          = pa.top_k
category_names = pa.category_names
device         = pa.device



print('img_dir        : {}'.format(img_dir))
print('checkpoint     : {}'.format(checkpoint))
print('top_k          : {}'.format(top_k))
print('category_names : {}'.format(category_names))
print('GPU            : {}'.format(device))


cat_to_name=FunctionTool.label_mapping(category_names)
model = FunctionTool.load_model(checkpoint)

FunctionTool.predict(img_dir, model, top_k, device,cat_to_name)

print("all process done.")
