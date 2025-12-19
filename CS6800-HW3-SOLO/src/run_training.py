import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import torch
from src.dataset import BuildDataset, BuildDataLoader
from src.model import SOLO
from src.backbone import Resnet50Backbone


if __name__ == "__main__":
    # set seed
    torch.random.manual_seed(1)

    # load the data into data.Dataset
    dataset = BuildDataset("hw3 solo datasets")

    # random split the dataset into training and testset
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])
    # push the randomized training data into the dataloader

    batch_size = 8
    train_loader = BuildDataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = BuildDataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # images:         (batch_size, 3, 800, 1088)
    # labels:         list with len: batch_size, each (n_obj,)
    # masks:          list with len: batch_size, each (n_obj, 1, 800, 1088), Added channels dimention to work with torchvision functions
    # bounding_boxes: list with len: batch_size, each (n_obj, 4)

    backbone = Resnet50Backbone(use_pretrained=False)
    model = SOLO(backbone=backbone)
    logger = TensorBoardLogger("tensorboard", name="solo")
    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="solo_epoch={epoch:02d}",
        every_n_epochs=1,
        save_top_k=-1,
        save_weights_only=True,
    )
    trainer = pl.Trainer(
        max_epochs=40,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        accelerator="auto",
        devices=1,
    )

    with torch.autograd.set_detect_anomaly(True):
        trainer.fit(model, train_loader, test_loader)
