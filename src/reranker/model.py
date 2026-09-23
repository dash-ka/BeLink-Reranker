import os
from typing import Optional

from swift.llm import PtEngine, get_template

from .download import get_model_path

DEFAULT_RESPONSE_PREFIX = "<think>\n\n</think>\n\nAnswer"


class BeLinkReranker:
    """
    Thin wrapper around a swift PtEngine + template, configured for
    the biomedical entity linking / reranking task.
    """

    def __init__(self, engine, template):
        self.engine = engine
        self.template = template

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        response_prefix: str = DEFAULT_RESPONSE_PREFIX,
        hf_token: Optional[str] = None,
    ):
        """
        Loads the model + tokenizer/template via swift's PtEngine.

        model_path: local path to weights. If None, downloads from
                    the Hugging Face repo via download.get_model_path().
        hf_token: Hugging Face token, only needed if the model repo is private.
        """

        if model_path is None:
            model_path = get_model_path(token=hf_token)

        engine = PtEngine(model_path, use_hf=True)

        template = get_template(
            engine.model_meta.template,
            engine.processor,
            default_system=None,
            response_prefix=response_prefix,
        )
        engine.default_template = template

        return cls(engine, template)
