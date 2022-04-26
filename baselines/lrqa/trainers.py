import torch
import torch.nn.functional as F

from transformers import Trainer


class GenerationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # noinspection PyUnresolvedReferences
        assert self.label_smoother is None  # not yet supported, can add back if needed
        batch_size, num_options, seq_length = inputs["input_ids"].shape

        # We use a heuristic: return_output=True -> Prediction, else -> training
        # The reason is that for training, we don't use the negative examples
        if return_outputs:
            # Get outputs for all options
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            # Shift/slice tokens for LM scoring
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["input_ids"][..., 1:].contiguous()

            # Select only the option tokens, and only logprobs for the gold tokens per options
            mask = get_option_token_pred_mask_all(inputs)
            option_token_log_probs = F.log_softmax(shift_logits, dim=-1) * mask.unsqueeze(-1)
            scored_token_log_probs = torch.gather(
                option_token_log_probs,
                dim=-1,
                index=shift_labels.unsqueeze(-1)
            ).squeeze(-1)
            option_log_probs = scored_token_log_probs.sum(-1)
            choice_loss = F.cross_entropy(option_log_probs, inputs["labels"])
            outputs = {"loss": choice_loss, "logits": option_log_probs}
            return choice_loss, outputs
        else:
            # Only select the right option
            index_range = range(len(inputs["input_ids"]))
            correct_input_ids = inputs["input_ids"][index_range, inputs["labels"]]
            correct_attention_mask = inputs["attention_mask"][index_range, inputs["labels"]]
            outputs = model(
                input_ids=correct_input_ids,
                attention_mask=correct_attention_mask,
            )

            # Shift/slice tokens for LM scoring
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = correct_input_ids[..., 1:].contiguous()

            # Flatten the tokens
            unreduced_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            )
            unreduced_loss = unreduced_loss.view(batch_size, (seq_length - 1))

            # Compute loss
            mask = get_option_token_pred_mask_only_correct(inputs)
            loss = (unreduced_loss * mask).sum() / mask.sum()
            return loss


def get_option_token_pred_mask_all(inputs):
    """
    Get mask for relevant tokens for PREDICTIONS (shifted by 1 compared to inputs)
    for all options
    """
    batch_size, num_options, seq_length = inputs["input_ids"].shape
    mask = torch.zeros(batch_size, num_options, (seq_length-1)).bool()
    for i in range(batch_size):
        for j in range(num_options):
            #  preds will be back-shifted by one
            mask[i, j, inputs["option_token_start_idx"][i, j]-1:] = 1
            mask[i, j, inputs["option_token_end_idx"][i, j]-1:] = 0
    mask = mask.to(inputs["input_ids"].device)
    return mask


def get_option_token_pred_mask_only_correct(inputs):
    """
    Get mask for relevant tokens for PREDICTIONS (shifted by 1 compared to inputs)
    only for correct option
    """
    batch_size, num_options, seq_length = inputs["input_ids"].shape
    mask = torch.zeros(batch_size, (seq_length-1)).bool()
    for i in range(batch_size):
        correct_idx = inputs["labels"][i]
        #  preds will be back-shifted by one
        mask[i, inputs["option_token_start_idx"][i, correct_idx]-1:] = 1
        mask[i, inputs["option_token_end_idx"][i, correct_idx]-1:] = 0
    mask = mask.to(inputs["input_ids"].device)
    return mask
