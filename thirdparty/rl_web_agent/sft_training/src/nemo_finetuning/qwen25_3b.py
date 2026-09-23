import lightning as pl
import nemo_run as run
from nemo.collections.llm.gpt.model.qwen2 import Qwen2Model, Qwen25Config3B


@run.cli.factory(name="qwen25_3b")
def model(tensor_model_parallel_size: int = 1) -> run.Config[pl.LightningModule]:
    """
    Factory function to create a Qwen2.5 3b model configuration.

    Returns:
        run.Config[pl.LightningModule]: Configuration for the Qwen2.5 3b model.

    Examples:
        CLI usage:
            $ nemo llm pretrain model=qwen25_3b ...

        Python API usage:
            >>> model_config = model()
            >>> print(model_config)
    """

    return run.Config(Qwen2Model, config=run.Config(Qwen25Config3B, tensor_model_parallel_size=tensor_model_parallel_size))
