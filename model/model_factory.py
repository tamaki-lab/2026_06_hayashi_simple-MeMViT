from model import (
    ClassificationBaseModel,
    ModelConfig,
    MemViT,
    TimeSformer,

)


def configure_model(
        model_info: ModelConfig
) -> ClassificationBaseModel:
    """model factory

    model_info:
        model_info (ModelInfo): information for model

    Raises:
        ValueError: invalide model name given by command line

    Returns:
        ClassificationBaseModel: model
    """

    model_name = model_info.model_name
    if model_name == 'timesformer':
        model = TimeSformer(model_info.cfg)  # type: ignore[assignment]

    elif model_name == 'memvit':
        model = MemViT(model_info.cfg)  # type: ignore[assignment]

    else:
        raise ValueError('invalid model_info.model_name')

    return model
