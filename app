from flask import Flask, jsonify, request , redirect 
from llama_index.core import VectorStoreIndex, ServiceContext, Document, Settings
from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.llms.groq import Groq
import markdown
import os
from cnn import CNN_NeuralNet
import torch
from torchvision.transforms import transforms
from PIL import Image
from werkzeug.utils import secure_filename
import pandas as pd

disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple:healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']
disease_names = {
    'Apple___Apple_scab': 'Apple Scab',
    'Apple___Black_rot': 'Apple Black Rot',
    'Apple___Cedar_apple_rust': 'Cedar Apple Rust',
    'Apple:healthy': 'Healthy Apple',
    'Blueberry___healthy': 'Healthy Blueberry',
    'Cherry_(including_sour)___Powdery_mildew': 'Cherry Powdery Mildew',
    'Cherry_(including_sour)___healthy': 'Healthy Cherry',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora Leaf Spot',
    'Corn_(maize)___Common_rust_': 'Common Rust',
    'Corn_(maize)___Northern_Leaf_Blight': 'Northern Leaf Blight',
    'Corn_(maize)___healthy': 'Healthy Corn',
    'Grape___Black_rot': 'Grape Black Rot',
    'Grape___Esca_(Black_Measles)': 'Grape Esca (Black Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight (Isariopsis Leaf Spot)',
    'Grape___healthy': 'Healthy Grape',
    'Orange___Haunglongbing_(Citrus_greening)': 'Orange Huanglongbing (Citrus Greening)',
    'Peach___Bacterial_spot': 'Peach Bacterial Spot',
    'Peach___healthy': 'Healthy Peach',
    'Pepper,_bell___Bacterial_spot': 'Bell Pepper Bacterial Spot',
    'Pepper,_bell___healthy': 'Healthy Bell Pepper',
    'Potato___Early_blight': 'Potato Early Blight',
    'Potato___Late_blight': 'Potato Late Blight',
    'Potato___healthy': 'Healthy Potato',
    'Raspberry___healthy': 'Healthy Raspberry',
    'Soybean___healthy': 'Healthy Soybean',
    'Squash___Powdery_mildew': 'Squash Powdery Mildew',
    'Strawberry___Leaf_scorch': 'Strawberry Leaf Scorch',
    'Strawberry___healthy': 'Healthy Strawberry',
    'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
    'Tomato___Early_blight': 'Tomato Early Blight',
    'Tomato___Late_blight': 'Tomato Late Blight',
    'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
    'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Spider Mites (Two-Spotted Spider Mite)',
    'Tomato___Target_Spot': 'Tomato Target Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
    'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
    'Tomato___healthy': 'Healthy Tomato'
}

disease_model_path = 'plant_disease_model.pth'
disease_model = CNN_NeuralNet(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert('RGB')
    tr = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = disease_model(tr)
        predicted_index = output.argmax(dim=1).item()
        predicted_class = disease_classes[predicted_index]
        return predicted_class

app = Flask(__name__)

chat_engine = None

def init_models():
    """
    Initialize the LLM and embedding model settings.
    """
    model_name = "mixedbread-ai/mxbai-embed-large-v1"
    embed_model = MixedbreadAIEmbedding(
        'emb_07fb3b000f973e7486e883a7206bed120d45f9cb4c066329',
        model_name=model_name
    )
    llm = Groq(
        model="llama-3.1-8b-instant",
        api_key="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"
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

def initialize_chat_engine():
    """
    Initialize the chat engine from document data.
    """
    global chat_engine
    init_models()
    reader = SimpleDirectoryReader(input_files=['disease.pdf'])
    docs = reader.load_data()

    index = VectorStoreIndex.from_documents(docs)
    chat_engine = index.as_chat_engine(
        chat_mode="context",  
        verbose=True,
        system_prompt=("You are a chatbot, able to have normal interactions, as well as talk"
                        "about an essay discussing Paul Grahams life."
        ))
def format_response(text):
    """
    Convert the response text into HTML format using markdown.
    """
    html = markdown.markdown(text)
    return html

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    """
    Handles POST requests to the /chat endpoint.
    Expects a JSON body with a 'message' field.
    """
    user_input = request.json.get('message')
    print(user_input)
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    try:
        response = chat_engine.chat(user_input)
        formatted_response = format_response(response)
        return jsonify({"response": formatted_response}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
@app.route('/predict', methods=['GET', 'POST'])
def upload_image():
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
       
            predicted_disease = predict(file_path)
            out = disease_names[predicted_disease]
            output = f"The Detected image is of {disease_names[predicted_disease]}"
            
            prompt = f"Plant disease detected: {predicted_disease}. Provide management and treatment suggestions."
          
            engine_out = chat_engine.chat(prompt)
            formatted_output = markdown.markdown(engine_out)
            
    return jsonify({"response": formatted_output}), 200

if __name__ == '__main__':

    initialize_chat_engine()
    app.run(debug=True)