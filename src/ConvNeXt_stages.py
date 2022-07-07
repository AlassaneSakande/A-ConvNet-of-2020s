class ConvNeXt_stages(nn.Module):
  """
  Here is our stages class
  """
  def __init__(self, ResNet_block, stages_list):
    super().__init__()
    self.in_features = 96
    self.stage2 = self._stages(ResNet_block, stages_list[0], output_channel = 96)
    self.stage3 = self._stages(ResNet_block, stages_list[1], output_channel = 192)
    self.stage4 = self._stages(ResNet_block, stages_list[2], output_channel = 384)
    self.stage5 = self._stages(ResNet_block, stages_list[3], output_channel = 768)
  
  def forward(self, x):
    x = self.stage2(x)
    x = self.stage3(x)
    x = self.stage4(x)
    x = self.stage5(x)
    return x


  def _stages(self, ResNet_block, residual_blocks, output_channel, stride : int = 2):
    stages_list = []
    #output = self.in_features

    # Here is the downsampling layer, It will downsample the input by a factor = 2
    stages_list.append(nn.Sequential(
        nn.GroupNorm(num_groups=1, num_channels=self.in_features),
        nn.Conv2d(self.in_features, output_channel, kernel_size=2 , stride=2),
    )),*[stages_list.append(ResNet_block(output_channel, output_channel)) for i in range(residual_blocks - 1)]
     
    
    return nn.Sequential(*stages_list)
