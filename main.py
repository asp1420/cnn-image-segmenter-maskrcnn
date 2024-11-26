from lightning.pytorch.cli import LightningCLI
from modules.mrcnnmodule import MRCNNModule


def main() -> None:
    LightningCLI(
        model_class=MRCNNModule,
        save_config_kwargs={"overwrite": True}
    )


if __name__ == '__main__':
    main()
