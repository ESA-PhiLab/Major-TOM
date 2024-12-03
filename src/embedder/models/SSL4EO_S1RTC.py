import torch
from torchgeo.models import ResNet50_Weights
import timm
import numpy as np

class SSL4EO_S1RTC_Embedder(torch.nn.Module):
    '''
        This embedder uses as an SSL4EO Sentinel-1 Pre-trained Model.

        Project code: https://github.com/zhu-xlab/SSL4EO-S12

        Publication: https://arxiv.org/abs/2211.07044

    '''

    def __init__(self, s1_mean=[-12.54847273, -20.19237134], s1_std=[5.25697717,5.91150917]):
        '''
            s1_mean and s1_std used for normalizing the data
            values taken from the SSL4EO codebase
        '''
        super().__init__()

        self.s1_mean = torch.FloatTensor(s1_mean)
        self.s1_std = torch.FloatTensor(s1_std)

        # load model
        self.model = self.init_model()
        self.bands = ['vv','vh']
        self.size = 224,224

    def init_model(self):
        weights = ResNet50_Weights.SENTINEL1_ALL_MOCO
        model = timm.create_model('resnet50', in_chans=weights.meta['in_chans'])
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        model.fc=torch.nn.Identity()

        return model

    def normalize(self, img,scale=1.0):
        '''
            SAR signal normalization used for the SSL4EO models
        '''
        
        min_value = (self.s1_mean - 2 * self.s1_std).to(img.device)
        max_value = (self.s1_mean + 2 * self.s1_std).to(img.device)
        img = (img - min_value[:,None,None]) / (max_value - min_value)[:,None,None] * scale
        img = img.clip(0,scale).float()

        return img

    def preprocess(self, input):
       return self.normalize(input)

    def forward(self, input):
        return self.model(self.preprocess(input))