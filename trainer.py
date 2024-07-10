from lightning.pytorch.cli import LightningCLI
from lightning import LightningModule, LightningDataModule

def main():
    cli = LightningCLI(LightningModule, LightningDataModule, subclass_mode_model=True, subclass_mode_data=True, save_config_callback=None, parser_kwargs={"parser_mode": "omegaconf"})

if __name__ == "__main__":
    main()