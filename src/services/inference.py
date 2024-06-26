"""
This module is used for inference.
"""
import torch


class InferenceEngine:
    """
    This module is used for inference.
    """

    def __init__(
        self,
        context: str = None,
        question: str = None
    ):
        self.context = context,
        self.question = question,

    async def load_input(self, tokenizer):
        """_summary_

        Returns:
            _type_: _description_
        """
        inputs = tokenizer(
            self.question,
            self.context,
            return_tensors="pt"
        )
        return inputs

    async def load_output(self, model, inputs):
        """_summary_

        Returns:
        """

        with torch.no_grad():
            outputs = model(**inputs)
        print(type(outputs))
        return outputs

    async def get_answer_index(self, outputs):
        """_summary_

        Returns:
        """

        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        print(type(answer_start_index))
        return answer_start_index, answer_end_index

    async def get_predict_answer_tokens(self, inputs, answer_start_index, answer_end_index):
        """_summary_

        Args:
            inputs (_type_): _description_
            answer_start_index (_type_): _description_
            answer_end_index (_type_): _description_

        Returns:
            _type_: _description_
        """
        predict_answer_tokens = inputs.input_ids[0,
                                                 answer_start_index: answer_end_index + 1]

        return predict_answer_tokens

    async def decode_answer(self, tokenizer, predict_answer_tokens):
        """_summary_

        Args:
            tokenizer (_type_): _description_
            predict_answer_tokens (_type_): _description_

        Returns:
            _type_: _description_
        """

        predict_answer = tokenizer.decode(
            predict_answer_tokens,
            skip_special_tokens=True
        )

        return predict_answer
