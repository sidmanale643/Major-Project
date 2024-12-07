from flask import Flask, jsonify, request , redirect  , render_template
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

app = Flask(__name__)

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

def load_disease_model():
    model = CNN_NeuralNet(3, len(disease_classes))
    model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def init_chat_engine():
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

    reader = SimpleDirectoryReader(input_files=['disease.pdf'])
    docs = reader.load_data()

    index = VectorStoreIndex.from_documents(docs)
    return index.as_chat_engine(
        chat_mode="context",
        verbose=True,
        system_prompt=(
            "You are an agricultural chatbot, capable of having general conversations and offering detailed insights on farming techniques, crops, livestock, sustainability practices, and more. "
            "Additionally, you can engage in discussions about various topics, including an essay about Paul Graham's life, his contributions to technology, and his philosophical views. "
            "Feel free to provide helpful advice, answer questions, and offer thoughtful commentary on these subjects."
        )
    )

disease_model = load_disease_model()
chat_engine = init_chat_engine()

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
        return disease_classes[predicted_index]

def provide_disease_management(disease_name):
    prompt = f"Plant disease detected: {disease_name}. Provide management and treatment suggestions. If the name of the detected disease has 'healthy' in the name, do not provide treatment and suggestions."
    return chat_engine.chat(prompt)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_disease():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_path = f"uploads/{image.filename}"
    image.save(image_path)

    disease = predict(image_path)
    management = provide_disease_management(disease)

    return jsonify({'disease': disease, 'management': management})

if __name__ == '__main__':
    app.run(debug=True)
