from determined.pytorch import DataLoader, PyTorchTrial, PyTorchTrialContext
 
TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
 
class MyTrial(PyTorchTrial):
   def __init__(self, context: PyTorchTrialContext) -> None:
       self.context = context
   def build_training_data_loader(self) -> DataLoader:
       return DataLoader()
 
   def build_validation_data_loader(self) -> DataLoader:
       return DataLoader()
 
   def train_batch(self, batch: TorchData, epoch_idx: int, batch_idx: int)  -> Dict[str, Any]:
       return {}
 
   def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
       return {}