#the stem change to "Patchify" (7, 7) stride = 2 + MaxPool ---> (4, 4) stride = 4

class ConvNeXt_stem(nn.Sequential):
  """
  Initially in ResNet, the stem layer is composed by
  a (7, 7) Conv + stride = 2 following with a MaxPool layer
  We replace this with a single (4, 4) Conv + stride = 4 layer
  """
  def __init__(self, in_features: int, out_features: int, FULLCONV, **kwargs):
    super().__init__(
        nn.Conv2d(in_features, out_features, kernel_size = 4, stride = 4, **kwargs),
        nn.BatchNorm2d(out_features),
    )
