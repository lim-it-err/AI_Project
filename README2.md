# glasses Model 사용법 

# Demand Command for glasses Module
!pip install git+https://github.com/FrancescoSaverioZuppichini/glasses
!pip install huggingface_hub
!pip install torchinfo
!pip install rich
!pip install einops 

# ResNet Model Load 

from glasses.models import ResNet  
model = ResNet.resnet50(n_classes= 2) # n_classes means the number of output type , If u want to get pretrained model , just add pretrained = True in parameter. 


# Visualization , just example

from glasses.interpretability import GradCam, SaliencyMap
from torchvision.transforms import Normalize 

dataiter = iter(trainloader) 
images, labels = dataiter.next() 
plt.imshow(torchvision.utils.make_grid(images[0], normalize= True).permute(1,2,0)) 
_ = model.interpret(images[0].to(device).unsqueeze(0), using=GradCam()).show()