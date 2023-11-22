from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

def answer_question(question, context):
    # Load pre-trained BERT model and tokenizer fine-tuned on SQuAD
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')

    # Tokenize and encode input text
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract start and end scores from the model output
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the tokens with the highest start and end scores
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Get the answer span
    answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

# Example usage
context = "You are a chatbot of a library of college Pune Institute of Computer Techonolgy."
question = "What is college name?"

answer = answer_question(question, context)
print("Answer:", answer)
