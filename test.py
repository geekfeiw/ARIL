import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time
import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import time

from tqdm import tqdm


batch_size = 512


data_amp = sio.loadmat('data/test_data_split_amp.mat')
test_data_amp = data_amp['test_data']
test_data = test_data_amp
# data_pha = sio.loadmat('data/test_data_split_pha.mat')
# test_data_pha = data_pha['test_data']
# test_data = np.concatenate((test_data_amp,test_data_pha), 1)

test_activity_label = data_amp['test_activity_label']
test_location_label = data_amp['test_location_label']
test_label = np.concatenate((test_activity_label, test_location_label), 1)

num_test_instances = len(test_data)

test_data = torch.from_numpy(test_data).type(torch.FloatTensor)
test_label = torch.from_numpy(test_label).type(torch.LongTensor)
# test_data = test_data.view(num_test_instances, 1, -1)
# test_label = test_label.view(num_test_instances, 2)

test_dataset = TensorDataset(test_data, test_label)
test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

aplnet = torch.load('weights/net1111_Train100.0Test88.129Train99.910Test95.683.pkl')
aplnet = aplnet.cuda().eval()


correct_test_loc = 0
correct_test_act = 0

for i, (samples, labels) in enumerate(test_data_loader):
    with torch.no_grad():
        samplesV = Variable(samples.cuda())
        labels_act = labels[:, 0].squeeze()
        labels_loc = labels[:, 1].squeeze()
        labelsV_act = Variable(labels_act.cuda())
        labelsV_loc = Variable(labels_loc.cuda())

        predict_label_act, predict_label_loc,_,_,_,_,_,_,_ = aplnet(samplesV)

        # for tsne visualization
        # act1, loc1, x, c1, c2, c3, c4, act, loc = aplnet(samplesV)
        # sio.savemat('vis/fig_tsne/out_act_conf.mat', {'out_max': act1.view(act1.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_loc_conf.mat', {'out_max': loc1.view(loc1.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_maxpool.mat', {'out_max': x.view(x.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_c1.mat', {'out_max': c1.view(c1.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_c2.mat', {'out_max': c2.view(c2.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_c3.mat', {'out_max': c3.view(c3.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_c4.mat', {'out_max': c4.view(c4.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_fc_act.mat', {'out_max': act.view(act.shape[0], -1).cpu().numpy()})
        # sio.savemat('vis/fig_tsne/out_fc_loc.mat', {'out_max': loc.view(loc.shape[0], -1).cpu().numpy()})

        prediction = predict_label_act.data.max(1)[1]
        sio.savemat('vis/actResult.mat',{'act_prediction':prediction.cpu().numpy()})
        correct_test_act += prediction.eq(labelsV_act.data.long()).sum()
        print(correct_test_act.cpu().numpy()/num_test_instances)

        prediction = predict_label_loc.data.max(1)[1]
        sio.savemat('vis/locResult.mat', {'loc_prediction': prediction.cpu().numpy()})
        correct_test_loc += prediction.eq(labelsV_loc.data.long()).sum()
        print(correct_test_loc.cpu().numpy() / num_test_instances)

