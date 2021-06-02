import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR
from attrdict import AttrDict

from typing import Any, Dict, Sequence, Tuple, Union, cast
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from nts_net.config import PROPOSAL_NUM  # this is also used into another file
from nts_net import model

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
 
class MyTrial(PyTorchTrial):
   def __init__(self, context: PyTorchTrialContext) -> None:
       self.context = context

       self.model = self.context.wrap_model(model.attention_net(topN=PROPOSAL_NUM, num_classes=len(classes_dict)))
       self.hparams = AttrDict(self.context.get_hparams())

       # define optimizers
       raw_parameters = list(self.model.pretrained_model.parameters())
       part_parameters = list(self.model.proposal_net.parameters())
       concat_parameters = list(self.model.concat_net.parameters())
       partcls_parameters = list(self.model.partcls_net.parameters())

       self.raw_optimizer = self.context.wrap_optimizer(torch.optim.SGD(raw_parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))
       self.concat_optimizer = self.context.wrap_optimizer(torch.optim.SGD(concat_parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))
       self.part_optimizer = self.context.wrap_optimizer(torch.optim.SGD(part_parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))
       self.partcls_optimizer = self.context.wrap_optimizer(torch.optim.SGD(partcls_parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))


   def build_training_data_loader(self) -> DataLoader:
       transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
       trainset = torchvision.datasets.CIFAR10(root=self.download_directory, train=True, download=True, transform=transform)
       return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())
       #return DataLoader()
 
   def build_validation_data_loader(self) -> DataLoader:
       return DataLoader()
 
   def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
       return {}
 
   def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
       return {}