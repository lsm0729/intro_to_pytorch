import FunctionTool
import argparse
import sys

parser = argparse.ArgumentParser(description='build and train model!')




parser.add_argument('data_dir',action="store",type=str,
                    help='Enter path to dataset EX) ./local/flowers/')

parser.add_argument('--gpu', dest="device", action="store_true", default=False,
                    help='Turn GPU mode on or off, default is off.')

parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth",
                    help='Enter path to checkpoint.pth EX) ./local/checkpoint.pth ')

parser.add_argument('--arch', dest="arch", action="store", default="vgg13", type = str,
                    help='Slect architecture to use. Default is vgg13')

parser.add_argument('--lr', dest="lr", action="store", default=0.001,
                    help='Slect lr to use. Default is 0.001')

parser.add_argument('--hidden_units', dest="hidden_unit", action="store", default=4096,
                    help='Slect hidden_unit to use. Default is 4096')

parser.add_argument('--epochs', dest="epochs", action="store", default=2,
                    help='Slect epochs to use. Default is 2')

parser.add_argument('--dropout', dest="dropout", action="store", default=0.5,
                    help='Slect dropout to use. Default is 0.5')

pa = parser.parse_args()

data_dir = pa.data_dir
device   = pa.device
save_dir = pa.save_dir
arch     = pa.arch
lr       = pa.lr
hu       = pa.hidden_unit
dropout  = pa.dropout
epochs   = pa.epochs


print('data_dir    : {}'.format(data_dir))
print('gpu         : {}'.format(device))
print('save_dir    : {}'.format(save_dir))
print('arch        : {}'.format(arch))
print('lr          : {}'.format(lr))
print('hidden_unit : {}'.format(hu))
print('dropout     : {}'.format(dropout))
print('epochs      : {}'.format(epochs))

try:
    image_datasets,data_loaders= FunctionTool.data_preprocess(data_dir=data_dir)
except:
    print("wrong data directory. pleasse check again.")
    sys.exit(1)

model, criterion, optimizer= FunctionTool.build_model(device,dropout,hu,lr,arch)

model = FunctionTool.train_model(model,criterion,optimizer,device,data_loaders,epochs)

FunctionTool.save_chkpoint(model, save_dir,image_datasets,hu,lr,epochs,arch)

FunctionTool.test_model(model, device, data_loaders)



print("all process done.")
