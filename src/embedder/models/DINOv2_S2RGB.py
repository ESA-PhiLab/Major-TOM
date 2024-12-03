import torch
from transformers import AutoImageProcessor, AutoModel

class DINOv2_S2RGB_Embedder(torch.nn.Module):
    '''
        Embedding wrapper for DINOv2 and Sentinel-2 data.

        Preprocessing: Divide by 10,000 and multiply by 2.5 for a True-Colour Image, followed by DINOv@ processor

        Model: Takes RGB input of shape 224 x 224 and produces a
    '''

    def __init__(self):
        super().__init__()

        # load model
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')
        self.bands = ['B04', 'B03', 'B02'] # RGB
        self.size = self.processor.crop_size['height'], self.processor.crop_size['width']

    def normalize(self, input):
        '''
            Maps Sentinel-2 to True-Colour
        '''
        return (2.5 * (input / 1e4)).clip(0,1)

    def forward(self, input):
        model_input = self.processor(self.normalize(input), return_tensors="pt")
        outputs = self.model(model_input['pixel_values'].to(self.model.device))
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states.mean(dim=1).cpu()