import json
import re
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Detect the available device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the BERT model and tokenizer and move to the correct device
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)
# Load the JSON transcript file
def load_json(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)                           

def load_txt(file_path):
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def clean_text(text):
    """Clean the text: remove newlines and non-alphanumeric characters."""
    text = re.sub(r'\n', ' ', text)  # Replace newline with space
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

def split_into_sentences(text):
    """Split the punctuated transcript into sentences."""
    sentences = re.split(r'[.!?]', text)  # Split by end punctuation
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

def get_bert_embedding(sentence):
    """Get BERT embeddings for a given sentence."""
    inputs = tokenizer(sentence, return_tensors='pt', truncation=True, padding=True)
    # Move inputs to the same device as the model
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():  # Disable gradient calculation
        outputs = model(**inputs)
    # Return the embeddings as a NumPy array on CPU
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()

def match_sentence_to_timestamps_with_bert(punctuated_sentences, transcript_json):
    """Match sentences to their corresponding timestamps using BERT embeddings."""
    highlighted_sections = []

    # Precompute embeddings for the entire JSON transcript
    print("Precomputing BERT embeddings for transcript entries...")
    json_embeddings = []
    for idx, entry in enumerate(transcript_json):
        entry_text = clean_text(entry['text'])
        entry_embedding = get_bert_embedding(entry_text)
        json_embeddings.append((entry, entry_embedding))
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(transcript_json)} transcript entries.")

    print("Precomputed embeddings for all transcript entries.")

    # Generate BERT embeddings for all sentences in the punctuated transcript
    for idx, sentence in enumerate(punctuated_sentences):
        sentence_embedding = get_bert_embedding(sentence)

        best_match = None
        best_similarity = -1

        print(f"\nProcessing sentence {idx + 1}/{len(punctuated_sentences)}: '{sentence}'")

        for entry, entry_embedding in json_embeddings:
            # Calculate cosine similarity between the sentence and transcript entry
            similarity = cosine_similarity(sentence_embedding, entry_embedding)[0][0]

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = entry

        # Add the best-matched sentence to highlighted sections if similarity is above a threshold
        if best_match and best_similarity > 0.5:  # Adjust the threshold as needed
            start_time = best_match['offset'] / 1000
            end_time = (best_match['offset'] + best_match['duration']) / 1000

            # Define sections like "Introduction", "Main Content", "Conclusion"
            if idx < len(punctuated_sentences) * 0.1:
                section = "Introduction"
            elif idx > len(punctuated_sentences) * 0.9:
                section = "Conclusion"
            else:
                section = "Main Content"

            highlighted_sections.append({
                "start_time": float(start_time),  # Convert to native Python float
                "end_time": float(end_time),      # Convert to native Python float
                "text": sentence,
                "section": section,
                "confidence_score": float(best_similarity)  # Convert to native Python float
            })
            print(f"Matched to transcript text: '{best_match['text']}' (Similarity: {best_similarity:.2f})")
        else:
            print(f"Could not confidently match the sentence.")

    return highlighted_sections

def main():
    # Load the files
    transcript_json = load_json('transcript.json')
    transcript_txt = load_txt('transcript.txt')

    # Clean and split the punctuated transcript into sentences
    punctuated_sentences = split_into_sentences(transcript_txt)

    print(f"Loaded {len(punctuated_sentences)} sentences from the punctuated transcript.")
    print(f"Loaded {len(transcript_json)} entries from the JSON transcript.")

    # Generate highlights using BERT embeddings
    highlighted_sections = match_sentence_to_timestamps_with_bert(punctuated_sentences, transcript_json)

    # Print the highlighted sections
    print("\nHighlighted Sections:")
    for highlight in highlighted_sections:
        print(f"{highlight['start_time']:.2f}s - {highlight['end_time']:.2f}s: [{highlight['section']}] '{highlight['text']}' (Confidence: {highlight['confidence_score']:.2f})")

    # Save the highlights to a JSON file
    with open('highlights_with_bert.json', 'w', encoding='utf-8') as f:
        json.dump(highlighted_sections, f, indent=4)

    print("\nHighlights have been saved to 'highlights_with_bert.json'.")

if __name__ == '__main__':
    main()  
