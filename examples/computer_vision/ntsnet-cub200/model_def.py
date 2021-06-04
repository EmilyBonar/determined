import torch
import torchvision
from torch import nn
from torchvision import transforms
from attrdict import AttrDict
import tempfile

from typing import Any, Dict, Sequence, Tuple, Union, cast
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
from nts_net.config import PROPOSAL_NUM  # this is also used into another file
from nts_net import model

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
 
class MyTrial(PyTorchTrial):
   def __init__(self, context: PyTorchTrialContext) -> None:
       self.context = context

       self.download_directory = tempfile.mkdtemp()

       self.model = self.context.wrap_model(model.attention_net())

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
 
   def build_validation_data_loader(self) -> DataLoader:
       transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
       trainset = torchvision.datasets.CIFAR10(root=self.download_directory, train=False, download=True, transform=transform)
       return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())
 
   def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
       batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
       data, labels = batch

       output = self.model(data)
       loss = torch.nn.functional.nll_loss(output, labels)

       self.context.backward(loss)
       self.context.step_optimizer(self.raw_optimizer)
       self.context.step_optimizer(self.concat_optimizer)
       self.context.step_optimizer(self.part_optimizer)
       self.context.step_optimizer(self.partcls_optimizer)
       return {"loss": loss}
 
   def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
       batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
       data, labels = batch

       output = self.model(data)
       validation_loss = torch.nn.functional.nll_loss(output, labels).item()

       pred = output.argmax(dim=1, keepdim=True)
       accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)

       return {"validation_loss": validation_loss, "accuracy": accuracy}