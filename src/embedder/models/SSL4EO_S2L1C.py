import torch
from torchgeo.models import ResNet50_Weights
import timm

class SSL4EO_S2L1C_Embedder(torch.nn.Module):
    '''
        This embedder uses as an SSL4EO Sentinel-2 Pre-trained Model.

        Project code: https://github.com/zhu-xlab/SSL4EO-S12

        Publication: https://arxiv.org/abs/2211.07044

    '''


    def __init__(self):
        super().__init__()

        # load model
        self.model = self.init_model()
        self.bands = ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
        self.size = 224,224

    def init_model(self):
        weights = ResNet50_Weights.SENTINEL2_ALL_DINO
        model = timm.create_model('resnet50', in_chans=weights.meta['in_chans'])
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        model.fc=torch.nn.Identity()

        return model

    def preprocess(self, input):
       return input / 1e4

    def forward(self, input):
        return self.model(self.preprocess(input))