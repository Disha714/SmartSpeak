from transformers import pipeline

# Load the summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Read input text from output_transcript.txt
with open("output_transcript.txt", "r") as f:
    text = f.read()

# Summarize
summary = summarizer(text, max_length=100, min_length=5, do_sample=False)[0]['summary_text']

# Print summary
print("\n--- Summary ---\n", summary)

# Save to file (optional)
with open("summary_output.txt", "w") as f:
    f.write(summary)
