# class for creating a simple Conv + Normalization + Activation

class FULLCONV(nn.Sequential):
  """
  We use nn.Sequential here rather than nn.Module
  because we won't have any method in this class
  """
  def __init__(self, in_channel: int, out_channel: int, kernel_size: int, norm = nn.BatchNorm2d, activation = nn.ReLU, **kwargs):
    """
    in_channel : The input feature size of the convolution
    out_channel : THe ouput feature size of the convolution
    kenel_size : size of the kernel (3 = (3, 3))
    norm : The normalization function, we use Batch Normalization for ResNet
    activation : The activation function, usually ReLU for ResNet
    """
    
    #Here we return a Conv + norm + act 
  
    super().__init__(
        nn.Conv2d(in_channel, out_channel, kernel_size = kernel_size, padding = kernel_size//2 , **kwargs),
        norm(out_channel),
        activation(),
    )
