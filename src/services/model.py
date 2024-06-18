"""_summary_
"""
from transformers import (AutoTokenizer,
                          AutoModelForQuestionAnswering)


class ModelLoader:
    def __init__(
        self,
        model_name: str = None,
        local_only: bool = True
    ):
        self.model_name = model_name,
        self.local_only = local_only

    def load_tokenizer(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        return tokenizer

    def load_model(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)

        return model
