from nltk.tokenize import TextTilingTokenizer

text = """
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. As such, NLP is related to the area of human-computer interaction. Many challenges in NLP involve natural language understanding. Recent advances in AI have led to significant progress in many areas of NLP.
Machine learning algorithms, particularly deep learning techniques, have shown considerable success in processing human language for various tasks. However, significant challenges still remain, particularly in ensuring that models understand context and nuances in language.
"""

# Initialize the TextTiling tokenizer
tt_tokenizer = TextTilingTokenizer()

# Segment the text into topical segments
segments = tt_tokenizer.tokenize(text)

for i, segment in enumerate(segments):
    print(f"Segment {i+1}:\n{segment}\n")
