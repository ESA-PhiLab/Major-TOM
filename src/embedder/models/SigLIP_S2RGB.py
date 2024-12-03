from open_clip import create_model_from_pretrained, get_tokenizer
import torch

class SigLIP_S2RGB_Embedder(torch.nn.Module):
    '''
        Embedding wrapper for SigLIP and Sentinel-2 data.

        Preprocessing: Divide by 10,000 and multiply by 2.5 for a True-Colour Image, followed by DINOv@ processor

        Model: Takes RGB input of shape 384 x 384 and produces a
    '''

    def __init__(self):
        super().__init__()

        # load model
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        self.bands = ['B04', 'B03', 'B02']
        self.size = self.preprocess.transforms[0].size

    def normalize(self, input):
        '''
            Maps Sentinel-2 to True-Colour
        '''
        return (2.5 * (input / 1e4)).clip(0,1)

    def forward(self, input):
        preprocess_input = self.normalize(input)

        # normalization only
        model_input = self.preprocess.transforms[-1](preprocess_input)
        
        return self.model.encode_image(model_input)