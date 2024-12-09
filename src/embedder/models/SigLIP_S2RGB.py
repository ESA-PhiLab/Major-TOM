from open_clip import create_model_from_pretrained, get_tokenizer
import torch

class SigLIP_S2RGB_Embedder(torch.nn.Module):
    """
    Embedding wrapper for SigLIP and Sentinel-2 data.

    This model processes Sentinel-2 RGB data and embeds it into a feature space using the DINOv@ transformer model.
    The preprocessing includes normalizing Sentinel-2 values to create a True-Colour image before passing it through
    the model. The final output is a high-dimensional feature vector representing the input image.

    Preprocessing:
        - Sentinel-2 bands are divided by 10,000 to scale the reflectance values.
        - Then, the values are multiplied by 2.5 to map them into the [0, 1] range for True-Colour images.
        - The model input is further processed using the DINOv@ preprocessor.

    Model:
        - Takes an RGB input of shape 384x384 pixels and produces an embedding vector.
    """

    def __init__(self):
        super().__init__()

        # load model
        self.model, self.preprocess = create_model_from_pretrained('hf-hub:timm/ViT-SO400M-14-SigLIP-384')
        # Sentinel-2 RGB bands (B04 - Red, B03 - Green, B02 - Blue)
        self.bands = ['B04', 'B03', 'B02']
        self.size = self.preprocess.transforms[0].size

    def normalize(self, input):
        """
        Normalizes Sentinel-2 image data to create a True-Colour image.

        Sentinel-2 images are scaled to reflectance values in the range [0, 1]. This function:
        - Divides the input by 10,000 to scale Sentinel-2 values.
        - Multiplies the result by 2.5 to map the values into the True-Colour image range.

        Args:
            input (torch.Tensor or np.ndarray): Input image with Sentinel-2 reflectance values.

        Returns:
            torch.Tensor: Normalized True-Colour image, clipped to the range [0, 1].
        """
        return (2.5 * (input / 1e4)).clip(0,1)

    def forward(self, input):
        """
        Forward pass through the SigLIP model.

        This method normalizes the input Sentinel-2 image to a True-Colour representation and processes it through
        the model to obtain an embedding.

        Args:
            input (torch.Tensor): A Sentinel-2 image, typically of shape (C, H, W), where C=3 (RGB), 
                                  H=384, and W=384.

        Returns:
            torch.Tensor: The image embedding produced by the model.
        """
        preprocess_input = self.normalize(input)

        # normalization only
        model_input = self.preprocess.transforms[-1](preprocess_input)
        
        return self.model.encode_image(model_input)