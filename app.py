from flask import Flask, render_template, request, jsonify
from chatbot import FAQChatbot

app = Flask(__name__,template_folder='.')
bot = FAQChatbot()

@app.route('/')
def home():
    return render_template('chat.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.get_json().get('message', '')
    return jsonify(bot.respond(user_message))

if __name__ == "__main__":
    app.run(debug=True, port=5000)