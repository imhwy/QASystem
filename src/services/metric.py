"""
"""

import datasets
import evaluation
import collections
from tqdm.auto import tqdm


class Metric:
    """
    """

    def __init__(
        self,
        trained_checkpoint,
        tokenizer,
        model,
        validation_dataset,
        max_length,
        stride
    ):
        self.trained_checkpoint = trained_checkout,
        self.tokenizer = tokenizer,
        self.model = model,
        self.validation_dataset = validation_dataset,
        self.max_length = max_length,
        self.stride = stride,
        self.squad_metric = evaluate.load("squad")

    def preprocess_validation_dataset(self):
        """
        """

        questions = [q.strip() for q in self.validation_dataset["question"]]
        inputs = self.tokenizer(
            questions,
            self.validation_dataset["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_map = inputs.pop("overflow_to_sample_mapping")
        example_ids = []

        for i in range(len(inputs["input_ids"])):
            sample_idx = sample_map[i]
            example_ids.append(self.validation_dataset["id"][sample_idx])

            sequence_ids = inputs.sequence_ids(i)
            offset = inputs["offset_mapping"][i]
            inputs["offset_mapping"][i] = [
                o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
            ]

        inputs["example_id"] = example_ids
        return inputs

    def compute_metrics(self, start_logits, end_logits, features, examples):
        """
        """

        example_to_features = collections.defaultdict(list)
        for idx, feature in enumerate(features):
            example_to_features[feature["example_id"]].append(idx)

        predicted_answers = []
        for example in tqdm(examples):
            example_id = example["id"]
            context = example["context"]
            answers = []

            # Loop through all features associated with that example
            for feature_index in example_to_features[example_id]:
                start_logit = start_logits[feature_index]
                end_logit = end_logits[feature_index]
                offsets = features[feature_index]["offset_mapping"]

                start_indexes = np.argsort(
                    start_logit)[-1: -n_best - 1: -1].tolist()
                end_indexes = np.argsort(
                    end_logit)[-1: -n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                best_answer = max(answers, key=lambda x: x["logit_score"])
                predicted_answers.append(
                    {"id": example_id, "prediction_text": best_answer["text"]}
                )
            else:
                predicted_answers.append(
                    {"id": example_id, "prediction_text": ""})

        theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in examples]
        return theoretical_answers, predicted_answers

    def compute_squad_metric(self, theoretical_answers, predicted_answers):
        """
        """

        result = self.squad_metric.compute(
            predictions=predicted_answers,
            references=theoretical_answers
        )
        return result
