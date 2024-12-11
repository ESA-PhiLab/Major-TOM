import torch
from torchgeo.models import ResNet50_Weights
import timm

class SSL4EO_S2L1C_Embedder(torch.nn.Module):
    """
    SSL4EO Embedder for Sentinel-2 data using a pre-trained model.

    This model is based on the SSL4EO (Self-Supervised Learning for Earth Observation) approach,
    using a pre-trained ResNet50 model for Sentinel-2 data. The model is fine-tuned for Sentinel-2 
    images and can be used directly for feature extraction.

    Project Code:
        https://github.com/zhu-xlab/SSL4EO-S12

    Publication:
        https://arxiv.org/abs/2211.07044
    """



    def __init__(self):
        """
        Initializes the SSL4EO_S2L1C_Embedder by loading the pre-trained SSL4EO model.

        The model uses ResNet50 architecture, adapted for Sentinel-2 data with a specific
        weight configuration (`ResNet50_Weights.SENTINEL2_ALL_DINO`) provided by `torchgeo`.
        It also defines the bands used for Sentinel-2 data and sets the input image size to 
        224x224 pixels (the model input size).

        Attributes:
            model (torch.nn.Module): The ResNet50 model with pre-trained weights for Sentinel-2 data.
            bands (list): List of Sentinel-2 bands used for input data.
            size (tuple): The input image size expected by the model, set to 224x224 pixels.
        """
        super().__init__()

        # Load the pre-trained SSL4EO ResNet50 model
        self.model = self.init_model()

        # Define the Sentinel-2 L1C bands (e.g., B01, B02, B03, etc.)
        self.bands = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 
            'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
        ]

        # Define the expected input size of the model
        self.size = 224, 224

    def init_model(self):
        """
        Initializes the ResNet50 model with pre-trained weights for Sentinel-2 data.

        The model is loaded using the `timm` library, with Sentinel-2 specific weights 
        (`ResNet50_Weights.SENTINEL2_ALL_DINO`). The fully connected layer (`fc`) is replaced 
        with an identity function to obtain embeddings directly from the last convolutional 
        layer.

        Returns:
            torch.nn.Module: The initialized ResNet50 model.
        """
        weights = ResNet50_Weights.SENTINEL2_ALL_DINO
        model = timm.create_model('resnet50', in_chans=weights.meta['in_chans'])
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        model.fc=torch.nn.Identity()

        return model

    def preprocess(self, input):
        """
        Preprocesses the Sentinel-2 input data for the model.

        This function normalizes the input image by dividing the pixel values by 10,000. 
        This scaling step ensures that the reflectance values are mapped into an appropriate 
        range for the model.

        Args:
            input (torch.Tensor): Input image with Sentinel-2 reflectance values (e.g., shape: [C, H, W]).

        Returns:
            torch.Tensor: Preprocessed input, scaled by a factor of 10,000.
        """
       return input / 1e4

    def forward(self, input):
        """
        Forward pass through the model.

        The input image is preprocessed and then passed through the ResNet50 model to obtain the embedding.

        Args:
            input (torch.Tensor): Preprocessed Sentinel-2 image (e.g., shape: [C, H, W]).

        Returns:
            torch.Tensor: The output embedding from the model.
        """
        return self.model(self.preprocess(input))