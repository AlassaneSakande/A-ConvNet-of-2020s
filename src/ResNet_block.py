class ResNet_block(nn.Module):
  """
  A class for returning the bottleneck
  bottleneck in ResNet is usually composed by:
  main_path:()
    - (7, 7) Conv + Normalization 
    - (1, 1) Conv + Normalization + Activation
    - (1, 1) Conv + Normalization
  short_path : (1, 1) Conv + Normalization
    ResNet_block = main_path + shortcut
  """
  def __init__(self, in_features: int, out_features: int, cardinality: int = 4, stride: int = 1,
              drop_p: float = .0,
              layer_scaler_init_value: float = 1e-6,):
    """
    in_features = The input feature size
    out_features = The output features size
    cardinality = Use to control the reduction and expansion of input and output size through a stage
    stride = Essentially used to shrink the kernel size
    """
    super(ResNet_block, self).__init__()
    expansion = out_features*cardinality

    self.main_path = nn.Sequential(
        nn.Conv2d(in_features, in_features, kernel_size=7, stride=stride, padding = 3, groups = in_features),
        # Layer Normalization
        
        nn.LayerNorm([3, 224, 244]),

        nn.Conv2d(in_features, expansion, kernel_size=1),
        #Apply the GELU
        nn.GELU(),

        nn.Conv2d(expansion, out_features, kernel_size=1), 
    )
    self.layer_scaler = LayerScaler(layer_scaler_init_value, out_features)
    self.drop_path = StochasticDepth(drop_p, mode="batch")


  def forward(self, x):
    residual = x
    x = self.main_path(x)
    x = self.layer_scaler(x)
    x = self.drop_path(x)
    x += residual
    return x
