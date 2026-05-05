import itertools
from collections import defaultdict
import random
import numpy
import torch

from encoders.TaskEncoder import TaskEncoder
from encoders.TokenizerFix import char_to_token_wrapper
import nltk
from nltk.corpus import brown


class QuestionAnswerEncoder(TaskEncoder):
    """Build prompts and aggregate token representations for QA-style tasks."""

    def __init__(self):
        """Initialize helper resources for prompt perturbations."""
        nltk.download('brown')
        words = list(brown.words())
        self.grouped_words = defaultdict(list)
        for word in words:
            self.grouped_words[len(word)].append(word)

    def random_words(self, sentence):
        """Replace each word with a random word of matching length.

        Args:
            sentence: Prompt text to randomize.
        """
        random_words = []

        for word in sentence[:-1].split():
            if len(word) in self.grouped_words:
                random_words.append(random.choice(self.grouped_words[len(word)]))
            else:
                random_words.append(random.choice(random.choice([words for length, words in self.grouped_words.items() if len(words) > 0])))

        random_sentence = " ".join(random_words)

        return random_sentence


    def get_instructions(self, task_types, entry, demonstrations=None, k=0, template_index=0, upfront=False, zero=False):
        """Construct the standard instruction sequence for one sample.

        Args:
            task_types: Prompt-template dictionary.
            entry: One dataset row.
            demonstrations: Optional dataframe for in-context examples.
            k: Number of demonstrations to sample.
            template_index: Prompt template index.
            upfront: Whether to use the upfront template family.
            zero: Whether to use the zero template family.
        """

        entry_task_type = entry["task_type"]
        if upfront:
            entry_task_type += "-upfront"
        if zero:
            entry_task_type += "-zero"

        template = task_types[entry_task_type]["templates"][template_index]

        context = entry["context"]
        question = entry["question"]

        instructions = []

        instructions.append({
            "role": "system",
            "content": template["system"]
        })

        if k > 0:
            demonstrations = demonstrations[demonstrations["context_id"] != entry["context_id"]]
            sampled_demonstrations = demonstrations.sample(k)
            for i, sampled_demonstration in sampled_demonstrations.iterrows():
                instructions.append({
                    "role": "user",
                    "content": template["user"].format(context=sampled_demonstration["context"], question=sampled_demonstration["question"])
                })
                instructions.append({
                    "role": "assistant",
                    "content": f" {sampled_demonstration['answer']}"
                })

        instructions.append({
            "role": "user",
            "content": template["user"].format(context=context, question=question)
        })

        return instructions


    def get_zero_instructions(self, task_types, entry, demonstrations=None, k=0, template_index=0, upfront=False, zero=False):
        """Construct zero-information prompts that contain only the sample text."""

        instructions = []
        context = entry["context"]

        instructions.append({
            "role": "system",
            "content": ""
        })

        if k > 0:
            demonstrations = demonstrations[demonstrations["context_id"] != entry["context_id"]]
            sampled_demonstrations = demonstrations.sample(k)
            for i, sampled_demonstration in sampled_demonstrations.iterrows():
                instructions.append({
                    "role": "user",
                    "content": sampled_demonstration["context"]
                })
                instructions.append({
                    "role": "assistant",
                    "content": f" {sampled_demonstration['answer']}"
                })

        instructions.append({
            "role": "user",
            "content": context
        })

        return instructions


    def get_zero_token_instructions(self, task_types, entry, demonstrations=None, k=0, template_index=0, upfront=False, zero=False):
        """Construct zero-information prompts that expose only the target span token."""
        instructions = []
        context = entry["context"]
        token = entry["spans"][0][0]

        instructions.append({
            "role": "system",
            "content": ""
        })

        if k > 0:
            demonstrations = demonstrations[demonstrations["context_id"] != entry["context_id"]]
            sampled_demonstrations = demonstrations.sample(k)
            for i, sampled_demonstration in sampled_demonstrations.iterrows():

                joined_span = "\n".join([span[0] for span in sampled_demonstration['spans']])
                instructions.append({
                    "role": "user",
                    "content": f"{joined_span}\n{sampled_demonstration['context']}"
                })
                instructions.append({
                    "role": "assistant",
                    "content": f" {sampled_demonstration['answer']}"
                })

        instructions.append({
            "role": "user",
            "content": f"{token}\n{context}"
        })

        return instructions
    def get_randomized_instructions(self, task_types, entry, demonstration = False, template_index=0, k=0, upfront=False, zero=False):
        """Construct prompts with randomized instruction wording."""

        entry_task_type = entry["task_type"]

        if upfront:
            entry_task_type += "-upfront"
        if zero:
            entry_task_type += "-zero"


        template = task_types[entry_task_type]["templates"][template_index]

        context = entry["context"]
        question = entry["question"]

        instructions = []

        instructions.append({
            "role": "system",
            "content": self.random_words(template["system"])
        })

        if demonstration:
            #todo
            pass

        user_instruction = template["user"].format(context=context, question=question)

        user_instruction = f"{self.random_words(user_instruction.split(context)[0])} {context} {self.random_words(user_instruction.split(context)[0])}"

        instructions.append({
            "role": "user",
            "content": user_instruction
        })

        return instructions

    def get_span_indices(self, encoded_batch, batch_frame, relevant_samples, tokenizer):
        """Map each annotated span onto token indices in the encoded prompt batch."""
        all_spans_token_indices = []

        for i, batch_element in batch_frame.iterrows():
            filtered_samples = relevant_samples[relevant_samples["instance_id"] == batch_element["instance_id"]]

            for _, row in filtered_samples.iterrows():

                input_string = row["context"]
                full_string = batch_element["compiled_instruction_text"]

                start_input_index = full_string.index(input_string)

                spans_token_indices = []
                for span, start_index, end_index in row["spans"]:
                    indices = []
                    for j in range(start_input_index + start_index, start_input_index + end_index):
                        indices.append(char_to_token_wrapper(encoded_batch, i, j))

                    indices = [ele for ele in indices if ele != None]

                    spans_token_indices.append(list(sorted(set(indices))))

                all_spans_token_indices.append(spans_token_indices)

        return all_spans_token_indices


    def get_question_indices(self, encoded_batch, batch_frame, relevant_samples, tokenizer):
        """Map each natural-language question onto token indices in the encoded prompt batch."""
        all_spans_token_indices = []

        for i, batch_element in batch_frame.iterrows():
            filtered_samples = relevant_samples[relevant_samples["instance_id"] == batch_element["instance_id"]]

            for _, row in filtered_samples.iterrows():

                question_string = row["question"]
                full_string = batch_element["compiled_instruction_text"]

                start_question_index = full_string.index(question_string)
                indices = []

                for j in range(len(question_string)):
                    indices.append(char_to_token_wrapper(encoded_batch, i, start_question_index+j))

                indices = [ele for ele in indices if ele != None]

                all_spans_token_indices.append(list(sorted(set(indices))))

        return all_spans_token_indices
    def get_scores(self, scores, encoded_batch, batch_frame, relevant_samples, tokenizer):
        """Attach generation scores to the matching dataset rows."""
        score_batch_elements = []

        for i, (idx, batch_element) in enumerate(batch_frame.iterrows()):
            filtered_samples = relevant_samples[relevant_samples["compiled_instruction_text"] == batch_element["compiled_instruction_text"]]

            for _, row in filtered_samples.iterrows():

                score_batch_element = row.copy()

                score_batch_element["scores"] = [list(ele[0].detach().cpu().numpy()) for ele in scores]

                score_batch_elements.append(score_batch_element)

        return score_batch_elements

    def get_sample_output_hidden_state(self, input_hidden_states, attention_caches, encoded_batch, batch_frame, relevant_samples, tokenizer):
        """Aggregate the two retained internal dump types: sample and output states."""
        aggregated_attention_caches = {key: torch.concat(values, dim=1) for key, values in attention_caches.items()}

        sample_encodings = []
        output_encodings = []

        for hidden_states, (i, batch_element) in zip(input_hidden_states, batch_frame.iterrows()):
            filtered_samples = relevant_samples[relevant_samples["instance_id"] == batch_element["instance_id"]]

            for _, row in filtered_samples.iterrows():

                input_string = row["context"]
                full_string = batch_element["compiled_instruction_text"]

                start_input_index = full_string.index(input_string)
                end_input_index = start_input_index + len(input_string)

                spans_token_indices = []
                for span, start_index, end_index in row["spans"]:
                    indices = []
                    for j in range(start_input_index + start_index, start_input_index + end_index + 1):
                        indices.append(char_to_token_wrapper(encoded_batch, i, j))

                    indices = [ele for ele in indices if ele != None]

                    spans_token_indices.append(list(sorted(set(indices))))

                spans_hidden_states = [
                    hidden_states[:,span_token_indices].mean(dim=1).detach().cpu()
                    for span_token_indices in spans_token_indices
                ]

                spans_hidden_states = torch.concat(spans_hidden_states, dim=1)

                for layer in range(spans_hidden_states.shape[0]):
                    row["layer"] = layer

                    if layer > 0:
                        output_layer_states = aggregated_attention_caches[f"o_layer-{layer}"][i].mean(dim=0).detach().cpu()
                        output_batch_element = row.copy()
                        output_batch_element["inputs_encoded"] = output_layer_states.numpy()
                        output_encodings.append(output_batch_element)

                    sample_batch_element = row.copy()
                    sample_batch_element["inputs_encoded"] = spans_hidden_states[layer].numpy()
                    sample_encodings.append(sample_batch_element)

        return sample_encodings, output_encodings

    def get_generated_hidden_state(self, output_hidden_states, encoded_batch, batch_frame, generated_texts, num_return_sequences):
        """Aggregate generation-side hidden states for each produced answer.

        Args:
            output_hidden_states: Decoder hidden states over generated tokens.
            encoded_batch: Tokenized prompt batch.
            batch_frame: Batch metadata as a dataframe.
            generated_texts: Decoded generated answers.
            num_return_sequences: Number of generations per input.
        """
        generated_encodings = []

        perspective_scores = {}#{
        #generated_text : self.get_perspective_attribute(generated_text)
        #for generated_text in generated_texts
        #}
        n_generated_tokens = output_hidden_states.shape[2]

        batch_size = batch_frame.shape[0]
        element_ids = list(itertools.chain.from_iterable([[element_id] *  num_return_sequences for element_id in batch_frame["element_id"]]))
        attention_masks = list(itertools.chain.from_iterable([[attention_mask] *  num_return_sequences for attention_mask in encoded_batch["attention_mask"]]))
        generation_ids = list(range(num_return_sequences)) * batch_frame.shape[0]

        for hidden_state, generated_text, instance_attention_mask, element_id, generation_id in zip(output_hidden_states, generated_texts, attention_masks , element_ids, generation_ids):
            generation_attention_mask = instance_attention_mask[-n_generated_tokens:]
            n_relevant_tokens = generation_attention_mask.sum()
            batch_element = batch_frame.query(f"element_id=={element_id}").iloc[0]
            generation_hidden_state = hidden_state[:,:n_relevant_tokens].mean(dim=1).detach().cpu()

            for layer, layer_state in enumerate(generation_hidden_state):
                layer_batch_element = batch_element.copy()

                layer_batch_element["element_id"] = element_id
                layer_batch_element["generation_id"] = generation_id
                layer_batch_element["layer"] = layer
                layer_batch_element["inputs_encoded"] = layer_state.numpy()
                layer_batch_element["generation_text"] = generated_text
                #layer_batch_element["sequence_score"] = float(sequence_score.detach().cpu())

                generated_encodings.append(layer_batch_element)

        return generated_encodings
