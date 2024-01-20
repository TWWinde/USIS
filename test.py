import models.models as models
import dataloaders.dataloaders as dataloaders
import util.utils as utils
import config
from util.fid_scores import fid_pytorch
from tqdm import tqdm

from util.metrics import metrics

generate_images = True
compute_fid_generation = False
generate_combined_images = False
compute_metrics = False



from util.utils import tens_to_im
import numpy as np
from torch.autograd import Variable

from collections import namedtuple


#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
_, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)
seg_saver = utils.seg_saver(opt)
image_saver_combine = utils.combined_images_saver(opt)
metrics_computer = metrics(opt, dataloader_val)
fid_computer = fid_pytorch(opt, dataloader_val)
#--- create models ---#
model = models.Unpaired_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

mae = []
mse = []
len_dataloader = len(dataloader_val)

if generate_images:
    #--- iterate over validation set ---#
    for i, data_i in tqdm(enumerate(dataloader_val)):
        mr_image, ct_image, label = models.preprocess_input(opt, data_i, test=True)
        generated = model(None, label, "generate", None).cpu().detach()
        seg_real = model(mr_image, None, "segment_real", None).cpu().detach()
        seg_fake = model(None, label, "segment_fake", None).cpu().detach()
        seg_saver(seg_real, seg_fake, data_i["name"])
        #image_saver(label, generated, data_i["name"])




if compute_metrics:
    metrics_computer.metrics_test(model)
    fid_computer.fid_test(model)


if generate_combined_images:
    # --- iterate over validation set ---#
    for i, data_i in tqdm(enumerate(dataloader_val)):

        label_save = data_i['label'].long()
        label_save = np.array(label_save).astype(np.uint8).squeeze(1)
        mr_image, ct_image, label = models.preprocess_input(opt, data_i, test=True)
        generated1 = model(None, label, "generate", None).cpu().detach()
        generated2 = model(None, label, "generate", None).cpu().detach()
        generated3 = model(None, label, "generate", None).cpu().detach()
        generated4 = model(None, label, "generate", None).cpu().detach()


        image_saver_combine(label, generated1, generated2, generated3, generated4, mr_image, ct_image, data_i["name"])





