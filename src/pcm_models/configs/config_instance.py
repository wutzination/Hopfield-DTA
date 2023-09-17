from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore

@dataclass
class Data:
    path: str = 'dummy'
    labels: str = 'dummy'
    mol_id: str = 'dummy'
    target_id: str = 'dummy'
    features: str = 'dummy'
    split: str = 'dummy'

@dataclass
class DataFnnHyper:
    path: str = 'dummy'
    affi_train: str = 'dummy'
    affi_val: str = 'dummy'
    std_mean: str = 'dummy'
    train_ecfp: str = 'dummy'
    val_ecfp: str = 'dummy'
    train_seq: str = 'dummy'
    val_seq: str = 'dummy'

@dataclass
class DataFnnTestSet:
    path: str = 'dummy'
    affi_train: str = 'dummy'
    affi_val: str = 'dummy'
    std_mean: str = 'dummy'
    train_ecfp: str = 'dummy'
    val_ecfp: str = 'dummy'
    train_seq: str = 'dummy'
    val_seq: str = 'dummy'

@dataclass
class DataHopfieldHyper:
    path: str = 'dummy'
    affi_train: str = 'dummy'
    affi_val: str = 'dummy'
    std_mean: str = 'dummy'
    train_ecfp: str = 'dummy'
    val_ecfp: str = 'dummy'
    train_seq: str = 'dummy'
    val_seq: str = 'dummy'

@dataclass
class Model:
    pass

@dataclass
class Training:
    pass

@dataclass
class Resources:
    pass

@dataclass
class Inference:
    pass

@dataclass
class MyConfig:
    data: Data = Data
    model: Model = Model
    training: Training = Training
    inference: Inference = Inference
    resources: Resources = Resources

@dataclass
class WurdingerConfigHyper:
    data: DataFnnHyper = DataFnnHyper
    model: Model = Model
    training: Training = Training
    inference: Inference = Inference
    resources: Resources = Resources

@dataclass
class WurdingerConfigTestSet:
    data: DataFnnTestSet = DataFnnTestSet
    model: Model = Model
    training: Training = Training
    inference: Inference = Inference
    resources: Resources = Resources

@dataclass
class WurdingerConfigHopfieldHyper:
    data: DataFnnTestSet = DataHopfieldHyper
    model: Model = Model
    training: Training = Training
    inference: Inference = Inference
    resources: Resources = Resources

