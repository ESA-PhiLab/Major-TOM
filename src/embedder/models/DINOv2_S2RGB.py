import torch
from transformers import AutoImageProcessor, AutoModel

class DINOv2_S2RGB_Embedder(torch.nn.Module):
    """
    Embedding wrapper for DINOv2 and Sentinel-2 data.

    This model uses the DINOv2 architecture to generate embeddings for Sentinel-2 RGB data. The input data (RGB bands) 
    is preprocessed by normalizing and mapping it to true-color values. Then, it is passed through the DINOv2 model 
    to obtain feature embeddings.

    Preprocessing:
        The input Sentinel-2 image is divided by 10,000 and multiplied by 2.5 to map it to a true-color image 
        (normalized to the range [0, 1]), followed by processing using the DINOv2 image processor.

    Model:
        The DINOv2 model processes RGB input images of shape [224, 224] and produces embeddings, which are then 
        averaged across the sequence dimension to obtain a fixed-size embedding vector.

    Model Components:
        - `AutoImageProcessor`: Preprocessing pipeline for handling Sentinel-2 data.
        - `AutoModel`: DINOv2 transformer model used for feature extraction.

    Attributes:
        processor (AutoImageProcessor): The DINOv2 image processor to handle preprocessing.
        model (AutoModel): The DINOv2 model used to generate embeddings from preprocessed images.
        bands (list): List of the Sentinel-2 bands used for RGB input (B04, B03, B02).
        size (tuple): The input size expected by the model (height, width) for the RGB image.
    """

    def __init__(self):
        """
        Initializes the DINOv2_S2RGB_Embedder by loading the pre-trained DINOv2 model and processor,
        and setting the expected input size for Sentinel-2 RGB data.

        This embedder uses the 'facebook/dinov2-base' model for feature extraction from Sentinel-2 
        true-color images (RGB).

        Attributes:
            processor (AutoImageProcessor): The DINOv2 image processor for preprocessing Sentinel-2 images.
            model (AutoModel): The pre-trained DINOv2 model for generating embeddings.
            bands (list): The Sentinel-2 bands used for RGB data (B04 - Red, B03 - Green, B02 - Blue).
            size (tuple): The expected input size of the image for the DINOv2 model (height, width).
        """
        super().__init__()

        # Load the DINOv2 processor and model from Hugging Face
        self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        self.model = AutoModel.from_pretrained('facebook/dinov2-base')

        # Define the RGB bands for Sentinel-2 (B04, B03, B02)
        self.bands = ['B04', 'B03', 'B02']

        # Extract the input size from the processor settings
        self.size = self.processor.crop_size['height'], self.processor.crop_size['width']


    def normalize(self, input):
        """
        Normalizes Sentinel-2 RGB data to true-color values.

        The input image (in raw Sentinel-2 reflectance values) is first divided by 10,000 to convert it 
        to reflectance values in the range [0, 1]. Then, the result is multiplied by 2.5 to obtain true-color 
        values that are suitable for input into the DINOv2 model.

        Args:
            input (torch.Tensor): The raw Sentinel-2 image tensor to be normalized.

        Returns:
            torch.Tensor: The normalized true-color image.
        """
        return (2.5 * (input / 1e4)).clip(0,1)

    def forward(self, input):
        """
        Forward pass through the model to generate embeddings for the input image.

        The input image is first normalized using the `normalize` method, then processed by the DINOv2 image processor 
        and passed through the DINOv2 model to generate embeddings. The output from the model is averaged across 
        the sequence dimension to obtain a fixed-size embedding.

        Args:
            input (torch.Tensor): The input Sentinel-2 image tensor with shape [C, H, W], where C=3 (RGB channels).

        Returns:
            torch.Tensor: The embedding vector, averaged over the sequence dimension, with shape [embedding_dim].
        """
        model_input = self.processor(self.normalize(input), return_tensors="pt")
        outputs = self.model(model_input['pixel_values'].to(self.model.device))
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states.mean(dim=1).cpu()