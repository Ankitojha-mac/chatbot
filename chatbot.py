import json, re, math
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

for pkg in ['punkt','punkt_tab','stopwords','wordnet']:
    nltk.download(pkg, quiet=True)

class FAQChatbot:
    def __init__(self, file="faqs.json"):
        self.lem = WordNetLemmatizer()
        self.sw = set(stopwords.words('english'))
        with open(file) as f: self.faqs = json.load(f)["faqs"]
        self.proc = [self.clean(f['question']) for f in self.faqs]
        self.vocab = set(t for p in self.proc for t in p)
        n = len(self.proc)
        self.idf = {t: math.log((n+1)/(sum(t in p for p in self.proc)+1))+1 for t in self.vocab}
        self.vecs = [self.tfidf(p) for p in self.proc]

    def clean(self, text):
        text = re.sub(r'[^\w\s]',' ', text.lower())
        return [self.lem.lemmatize(t) for t in word_tokenize(text) if t not in self.sw and len(t)>1]

    def tfidf(self, tokens):
        tf = Counter(tokens)
        n = len(tokens) or 1
        return {t:(c/n)*self.idf.get(t,1) for t,c in tf.items()}

    def cosine(self, a, b):
        dot = sum(a.get(t,0)*b.get(t,0) for t in set(a)|set(b))
        ma = math.sqrt(sum(x**2 for x in a.values()))
        mb = math.sqrt(sum(x**2 for x in b.values()))
        return dot/(ma*mb) if ma and mb else 0

    def respond(self, text):
        text = text.strip().lower()
        if text in ['hi','hello','hey','greetings']:
            return {"response":"Hello! 👋 Ask me anything!", "confidence":1.0, "matched":None}
        if text in ['bye','thanks','thank you','quit']:
            return {"response":"Goodbye! Have a great day! 😊", "confidence":1.0, "matched":None}

        vec = self.tfidf(self.clean(text))
        scores = [(i, self.cosine(vec, v)) for i,v in enumerate(self.vecs)]
        scores.sort(key=lambda x:x[1], reverse=True)
        best_idx, best_score = scores[0]

        if best_score < 0.15:
            return {"response":"Sorry, I don't have an answer for that. Try rephrasing!", "confidence":round(best_score,4), "matched":None}
        return {"response":self.faqs[best_idx]['answer'], "confidence":round(best_score,4), "matched":self.faqs[best_idx]['question']}


if __name__ == "__main__":
    bot = FAQChatbot()
    print("\n🤖 FAQ Chatbot Terminal Mode")
    print("Type 'quit' to exit\n")
    while True:
        q = input("You: ")
        if q.lower() in ['quit','exit']: break
        print(f"Bot: {bot.respond(q)['response']}\n")