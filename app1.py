from flask import Flask, request, jsonify
from langdetect import detect
from googletrans import Translator
import matplotlib.pyplot as plt
from huggingface_hub import login
from llama_cpp import Llama

# Login with your Hugging Face token
login(token="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")

# Initialize Flask app
app = Flask(__name__)

# Load the model using llama-cpp-python (for GGML model)
model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin"

# Download model manually if needed, or provide the correct path
model_path = "path_to_your_downloaded_model/llama-2-13b-chat.ggmlv3.q5_1.bin"

lcpp_llm = Llama(
    model_path=model_path,
    n_threads=2,  # Adjust based on CPU cores
    n_batch=512,  # Batch size (adjust based on system)
    n_gpu_layers=32  # Adjust for VRAM
)

# Translator for multilingual support
translator = Translator()

# Progress tracking dictionary
student_progress = {}

# Function to detect language
def detect_language(text):
    return detect(text)

# Translation function
def translate_text(text, target_lang="en"):
    return translator.translate(text, dest=target_lang).text

# Progress visualization
def visualize_progress(student):
    topics = student_progress.get(student, {})
    labels = list(topics.keys())
    values = list(topics.values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color="skyblue")
    plt.xlabel("Topics")
    plt.ylabel("Completion")
    plt.title(f"{student}'s Progress Overview")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Progress tracking function
def track_progress(student, topic):
    if student not in student_progress:
        student_progress[student] = {}
    student_progress[student][topic] = student_progress[student].get(topic, 0) + 1

# Function to generate AI response
def generate_response(prompt):
    response = lcpp_llm(
        prompt=prompt,
        max_tokens=1024,  # Adjust for detailed responses
        temperature=0.9,  # Creativity level
        top_p=0.95,  # Diversity in output
        repeat_penalty=1.2,
        top_k=100  # Token range restriction
    )
    return response["choices"][0]["text"].strip()

# API endpoint for the chatbot
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    student_name = data.get('student_name', '')
    inquiry = data.get('message', '')

    # Detect and translate if necessary
    language = detect_language(inquiry)
    if language != "en":
        inquiry = translate_text(inquiry)

    # Simplistic topic extraction
    topic = inquiry.split()[0]  # Improve this as needed

    # Generate the response
    support_response = generate_response(inquiry).strip()

    # Track progress
    track_progress(student_name, topic)

    # Prepare the final response
    final_response = {
        "inquiry": inquiry,
        "support_response": support_response
    }

    # Visualize progress
    visualize_progress(student_name)

    return jsonify(final_response)

if __name__ == '__main__':
    app.run(debug=True)
