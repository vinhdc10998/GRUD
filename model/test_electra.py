from transformers import ElectraForPreTraining, ElectraTokenizerFast
import torch

discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

sentence = "The quick brown fox jumps over the lazy dog"
fake_sentence = [
    "The quick brown fox ff over the lazy dog", 
    "The quick brown fox fake over the lazy dog"]

fake_tokens = tokenizer.tokenize(fake_sentence)
print(fake_tokens)
fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
print(fake_inputs, fake_inputs.shape)
discriminator_outputs = discriminator(fake_inputs)
print(discriminator_outputs[0])

predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

for token in fake_tokens:
    print(token)
for prediction in predictions.tolist():
    print(prediction)