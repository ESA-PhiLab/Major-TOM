import torch
from torchgeo.models import ResNet50_Weights
import timm
import numpy as np

class SSL4EO_S1RTC_Embedder(torch.nn.Module):
    """
    SSL4EO Embedder for Sentinel-1 data using a pre-trained model.

    This model is based on the SSL4EO (Self-Supervised Learning for Earth Observation) approach, 
    using a pre-trained ResNet50 model for Sentinel-1 radar data (SAR). The model is fine-tuned 
    to work with Sentinel-1 data and can be used directly for feature extraction.

    Project Code:
        https://github.com/zhu-xlab/SSL4EO-S12

    Publication:
        https://arxiv.org/abs/2211.07044
    """

    def __init__(self, s1_mean=[-12.54847273, -20.19237134], s1_std=[5.25697717,5.91150917]):
        """
        Initializes the SSL4EO_S1RTC_Embedder by setting up the mean and standard deviation for Sentinel-1 data normalization,
        and loading the pre-trained model.

        The model uses a pre-trained ResNet50 architecture adapted for Sentinel-1 radar (SAR) data, with weights provided 
        by the `torchgeo` library. The `s1_mean` and `s1_std` are used for normalizing the input data to the model.

        Args:
            s1_mean (list, optional): Mean values for Sentinel-1 radar (SAR) data. Default is set to SSL4EO's values.
            s1_std (list, optional): Standard deviation values for Sentinel-1 radar (SAR) data. Default is set to SSL4EO's values.

        Attributes:
            s1_mean (torch.FloatTensor): Mean values for normalization.
            s1_std (torch.FloatTensor): Standard deviation values for normalization.
            model (torch.nn.Module): The ResNet50 model initialized with pre-trained weights.
            bands (list): List of Sentinel-1 bands used for input data (VV, VH).
            size (tuple): The input size expected by the model (224x224 pixels).
        """
        super().__init__()

        self.s1_mean = torch.FloatTensor(s1_mean)
        self.s1_std = torch.FloatTensor(s1_std)

        # load model
        self.model = self.init_model()
        self.bands = ['vv','vh']
        self.size = 224,224

    def init_model(self):
        """
        Initializes the ResNet50 model with pre-trained weights for Sentinel-1 data.

        This method loads the pre-trained model weights for Sentinel-1 data from `ResNet50_Weights.SENTINEL1_ALL_MOCO` 
        and sets the fully connected layer (`fc`) to an identity function to output embeddings directly from the last 
        convolutional layer.

        Returns:
            torch.nn.Module: The initialized ResNet50 model.
        """
        weights = ResNet50_Weights.SENTINEL1_ALL_MOCO
        model = timm.create_model('resnet50', in_chans=weights.meta['in_chans'])
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)
        model.fc=torch.nn.Identity()

        return model

    def normalize(self, img,scale=1.0):
        """
        Normalizes the Sentinel-1 SAR (Synthetic Aperture Radar) data.

        This method normalizes the Sentinel-1 radar signals using the mean (`s1_mean`) 
        and standard deviation (`s1_std`) values. The radar data is normalized to a 
        standard range, and the pixel values are scaled using a factor (`scale`).

        Args:
            img (torch.Tensor): The input Sentinel-1 image to be normalized.
            scale (float, optional): The scaling factor for the normalized image. Default is 1.0.

        Returns:
            torch.Tensor: The normalized and scaled image.
        """
        
        
        min_value = (self.s1_mean - 2 * self.s1_std).to(img.device)
        max_value = (self.s1_mean + 2 * self.s1_std).to(img.device)
        img = (img - min_value[:,None,None]) / (max_value - min_value)[:,None,None] * scale
        img = img.clip(0,scale).float()

        return img

    def preprocess(self, input):
        """
        Preprocesses the Sentinel-1 SAR (Synthetic Aperture Radar) data before feeding it into the model.

        This method applies a logarithmic transformation to the input image to convert 
        it from linear scale to decibel (dB) scale. The image is clipped to avoid 
        logarithm of zero and then normalized using the `normalize` method.

        Args:
            input (torch.Tensor): The input Sentinel-1 image (e.g., VV or VH polarization).

        Returns:
            torch.Tensor: The preprocessed and normalized image in dB scale.
        """
        # Convert the input from linear scale to decibel (dB) scale
        dB_input = 10 * input.log10(input.clip(min=1e-10))  # Clip to prevent log(0)
    
        # Normalize the dB-scaled image
        return self.normalize(dB_input)

    def forward(self, input):
        """
        Forward pass through the model.

        The input image is preprocessed using the `preprocess` method and then passed 
        through the ResNet50 model to obtain an embedding.

        Args:
            input (torch.Tensor): Preprocessed Sentinel-1 image (e.g., shape: [C, H, W]).

        Returns:
            torch.Tensor: The output embedding from the model.
        """
        return self.model(self.preprocess(input))