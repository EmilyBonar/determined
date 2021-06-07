import torch
import torchvision
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms
from attrdict import AttrDict
from PIL import Image
import tempfile

from typing import Any, Dict, Sequence, Tuple, Union, cast
from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext, LRScheduler
from nts_net.config import PROPOSAL_NUM  # this is also used into another file
from nts_net import model

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
 
class MyTrial(PyTorchTrial):
   def __init__(self, context: PyTorchTrialContext) -> None:
       self.context = context

       self.download_directory = tempfile.mkdtemp()

       transform = transforms.Compose([
        # transforms.Resize((600, 600), Image.BILINEAR),
        # transforms.CenterCrop((448, 448)),
        transforms.Resize((448, 448), Image.BILINEAR),
        transforms.RandomHorizontalFlip(),  # solo se train
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
       ])
       self.trainset = torchvision.datasets.CIFAR10(root=self.download_directory, train=True, download=True, transform=transform)

       classes_dict = {i: v for i, v in enumerate(self.trainset.classes)}

       self.model = self.context.wrap_model(model.attention_net(topN=PROPOSAL_NUM, num_classes=len(classes_dict), device='cuda'))

       self.hparams = AttrDict(self.context.get_hparams())

       self.creterion = torch.nn.CrossEntropyLoss()

       # define optimizers
       raw_parameters = list(self.model.pretrained_model.parameters())
       part_parameters = list(self.model.proposal_net.parameters())
       concat_parameters = list(self.model.concat_net.parameters())
       partcls_parameters = list(self.model.partcls_net.parameters())

       self.raw_optimizer = self.context.wrap_optimizer(torch.optim.SGD(raw_parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))
       self.concat_optimizer = self.context.wrap_optimizer(torch.optim.SGD(concat_parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))
       self.part_optimizer = self.context.wrap_optimizer(torch.optim.SGD(part_parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))
       self.partcls_optimizer = self.context.wrap_optimizer(torch.optim.SGD(partcls_parameters, lr=self.hparams.lr, momentum=self.hparams.momentum, weight_decay=self.hparams.weight_decay))

       self.raw_scheduler = self.context.wrap_lr_scheduler(MultiStepLR(self.raw_optimizer, milestones=[60, 100], gamma=0.1), step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)
       self.concat_scheduler = self.context.wrap_lr_scheduler(MultiStepLR(self.concat_optimizer, milestones=[60, 100], gamma=0.1), step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)
       self.part_scheduler = self.context.wrap_lr_scheduler(MultiStepLR(self.part_optimizer, milestones=[60, 100], gamma=0.1), step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)
       self.partcls_scheduler = self.context.wrap_lr_scheduler(MultiStepLR(self.partcls_optimizer, milestones=[60, 100], gamma=0.1), step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH)


   def build_training_data_loader(self) -> DataLoader:
       return DataLoader(self.trainset, batch_size=self.context.get_per_slot_batch_size())
 
   def build_validation_data_loader(self) -> DataLoader:
       transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
       trainset = torchvision.datasets.CIFAR10(root=self.download_directory, train=False, download=True, transform=transform)
       return DataLoader(trainset, batch_size=self.context.get_per_slot_batch_size())
 
   def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
       batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
       data, labels = batch

       _, _, raw_logits, concat_logits, part_logits, _, top_n_prob = self.model(data)
       part_loss = model.list_loss(part_logits.view(len(data) * PROPOSAL_NUM, -1), labels.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(len(data), PROPOSAL_NUM)
       raw_loss = self.creterion(raw_logits, labels)
       concat_loss = self.creterion(concat_logits, labels)
       rank_loss = model.ranking_loss(top_n_prob, part_loss)
       partcls_loss = self.creterion(part_logits.view(len(data) * PROPOSAL_NUM, -1), labels.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))

       total_loss = raw_loss + rank_loss + concat_loss + partcls_loss

       self.context.backward(total_loss)
       self.context.step_optimizer(self.raw_optimizer)
       self.context.step_optimizer(self.concat_optimizer)
       self.context.step_optimizer(self.part_optimizer)
       self.context.step_optimizer(self.partcls_optimizer)
       return {"loss": total_loss}
 
   def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
       batch = cast(Tuple[torch.Tensor, torch.Tensor], batch)
       data, labels = batch

       output = self.model(data)
       validation_loss = torch.nn.functional.nll_loss(output, labels).item()

       pred = output.argmax(dim=1, keepdim=True)
       accuracy = pred.eq(labels.view_as(pred)).sum().item() / len(data)

       return {"validation_loss": validation_loss, "accuracy": accuracy}