# Load model directly
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained("chrlukas/stories-emotion-c8")
model = AutoModelForTokenClassification.from_pretrained("chrlukas/stories-emotion-c8")

