from model.deeplabv2 import Res_Deeplab
import argparse
import torch 
import torch.nn as nn
from torch.autograd import Variable
from data import get_data_path, get_loader
import numpy as np
import os 
from torch.utils import data

from PIL import Image
import scipy.misc
from utils.loss import CrossEntropy2d

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SSL evaluation script")
    parser.add_argument("-m","--model-path", type=str, default=None, required=True,
                        help="Model to evaluate")
    parser.add_argument("--gpu", type=int, default=(0,),
                        help="choose gpu device.")
    parser.add_argument("--save-output-images", action="store_true",
                        help="save output images")
    return parser.parse_args()

class CityscapeColorize(object):
    def __init__(self, n=19):
        self.cmap = color_map(19)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.shape
        color_image = np.zeros((3, size[0], size[1]), dtype=np.uint8)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image)
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        # handle void
        mask = (255 == gray_image)
        color_image[0][mask] = color_image[1][mask] = color_image[2][mask] = 255

        return color_image

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def evaluate(ignore_label=250):
    args = get_arguments()

    num_classes = 19
     
    model = Res_Deeplab(num_classes=num_classes)
    
    checkpoint = torch.load(args.model_path)
    try:
        model.load_state_dict(checkpoint['model'])
    except:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
        model.load_state_dict(checkpoint['model'])
    

    model.cuda()
    model.eval()

    num_classes = 19
    data_loader = get_loader('natural')
    data_path = get_data_path('natural')
    test_dataset = data_loader( data_path)
    testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    print('Evaluating, found ' + str(len(testloader))) 

    colorize = CityscapeColorize()

    for index, batch in enumerate(testloader):
        image, _, img_name = batch
        size = size[0]

        with torch.no_grad():
            interp = nn.Upsample(size=image.shape, mode='bilinear', align_corners=True)

            output  = model(Variable(image).cuda())
            output = interp(output)

            #label_cuda = Variable(label.long()).cuda()
            #criterion = CrossEntropy2d(ignore_label=ignore_label).cuda()  # Ignore label ??
            #loss = criterion(output, label_cuda)
            #total_loss.append(loss.item())

            output = output.cpu().data[0].numpy()
            print("point 1.")
            
            
            #gt = np.asarray(label[0].numpy(), dtype=np.int)

            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.int)
            #data_list.append([gt.reshape(-1), output.reshape(-1)])
            save_output_images = True
            save_dir = '/content/saved_img'
            if save_output_images:
                #filename = os.path.join(save_dir, '{}.png'.format(name[0]))
                save_name = img_name
                save_name = '/content/saved_img/' + save_name
                print("Filename:", save_name)
                color_file = Image.fromarray(colorize(output).transpose(1, 2, 0), 'RGB')
                #color_file.save(save_name)
            print("Processing...")
        if (index+1) % 100 == 0:
            print('%d processed'%(index+1))

if __name__=="__main__":
    print("Starting evaluation...")
    evaluate()