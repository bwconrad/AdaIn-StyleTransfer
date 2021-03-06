import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger
from argparse import Namespace
import os

from src.data import ContentStyleDataModule
from src.model import ASTModel

# Parse arguments
parser = LightningArgumentParser()
parser.add_lightning_class_args(pl.Trainer, None)
parser.add_lightning_class_args(ASTModel, "model")
parser.add_lightning_class_args(ContentStyleDataModule, "data")
parser.add_argument(
    "--output_path", type=str, help="Directory to save outputs.", default="output/"
)
parser.add_argument(
    "--experiment_name", type=str, help="Name of experiment.", default="default"
)
args = parser.parse_args()

# Define loggers
tb_logger = TensorBoardLogger(
    save_dir=args["output_path"], name=args["experiment_name"]
)

# Setup model
dm = ContentStyleDataModule(**args["data"])
model = ASTModel(**args["model"])
trainer = pl.Trainer.from_argparse_args(
    Namespace(**args),
    logger=tb_logger,
)

# Save config
parser.save(
    args, os.path.join(trainer.logger.experiment.log_dir, "config.yaml"), "yaml"
)

# Run trainer
trainer.tune(model, dm)
trainer.fit(model, dm)
