# import PIL
# from flask import Flask, jsonify, request , redirect 
# from llama_index.core import VectorStoreIndex, ServiceContext, Document, Settings
# from llama_index.embeddings.mixedbreadai import MixedbreadAIEmbedding
# from llama_index.core import SimpleDirectoryReader
# from llama_index.llms.groq import Groq
# import markdown
# import os
# from cnn import CNN_NeuralNet
# import torch
# from torchvision.transforms import transforms
# from PIL import Image
# from werkzeug.utils import secure_filename
# import pandas as pd

# chat_engine = None

# disease_classes = ['Apple___Apple_scab',
#                    'Apple___Black_rot',
#                    'Apple___Cedar_apple_rust',
#                    'Apple:healthy',
#                    'Blueberry___healthy',
#                    'Cherry_(including_sour)___Powdery_mildew',
#                    'Cherry_(including_sour)___healthy',
#                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
#                    'Corn_(maize)___Common_rust_',
#                    'Corn_(maize)___Northern_Leaf_Blight',
#                    'Corn_(maize)___healthy',
#                    'Grape___Black_rot',
#                    'Grape___Esca_(Black_Measles)',
#                    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#                    'Grape___healthy',
#                    'Orange___Haunglongbing_(Citrus_greening)',
#                    'Peach___Bacterial_spot',
#                    'Peach___healthy',
#                    'Pepper,_bell___Bacterial_spot',
#                    'Pepper,_bell___healthy',
#                    'Potato___Early_blight',
#                    'Potato___Late_blight',
#                    'Potato___healthy',
#                    'Raspberry___healthy',
#                    'Soybean___healthy',
#                    'Squash___Powdery_mildew',
#                    'Strawberry___Leaf_scorch',
#                    'Strawberry___healthy',
#                    'Tomato___Bacterial_spot',
#                    'Tomato___Early_blight',
#                    'Tomato___Late_blight',
#                    'Tomato___Leaf_Mold',
#                    'Tomato___Septoria_leaf_spot',
#                    'Tomato___Spider_mites Two-spotted_spider_mite',
#                    'Tomato___Target_Spot',
#                    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#                    'Tomato___Tomato_mosaic_virus',
#                    'Tomato___healthy']
# disease_names = {
#     'Apple___Apple_scab': 'Apple Scab',
#     'Apple___Black_rot': 'Apple Black Rot',
#     'Apple___Cedar_apple_rust': 'Cedar Apple Rust',
#     'Apple:healthy': 'Healthy Apple',
#     'Blueberry___healthy': 'Healthy Blueberry',
#     'Cherry_(including_sour)___Powdery_mildew': 'Cherry Powdery Mildew',
#     'Cherry_(including_sour)___healthy': 'Healthy Cherry',
#     'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Cercospora Leaf Spot',
#     'Corn_(maize)___Common_rust_': 'Common Rust',
#     'Corn_(maize)___Northern_Leaf_Blight': 'Northern Leaf Blight',
#     'Corn_(maize)___healthy': 'Healthy Corn',
#     'Grape___Black_rot': 'Grape Black Rot',
#     'Grape___Esca_(Black_Measles)': 'Grape Esca (Black Measles)',
#     'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': 'Grape Leaf Blight (Isariopsis Leaf Spot)',
#     'Grape___healthy': 'Healthy Grape',
#     'Orange___Haunglongbing_(Citrus_greening)': 'Orange Huanglongbing (Citrus Greening)',
#     'Peach___Bacterial_spot': 'Peach Bacterial Spot',
#     'Peach___healthy': 'Healthy Peach',
#     'Pepper,_bell___Bacterial_spot': 'Bell Pepper Bacterial Spot',
#     'Pepper,_bell___healthy': 'Healthy Bell Pepper',
#     'Potato___Early_blight': 'Potato Early Blight',
#     'Potato___Late_blight': 'Potato Late Blight',
#     'Potato___healthy': 'Healthy Potato',
#     'Raspberry___healthy': 'Healthy Raspberry',
#     'Soybean___healthy': 'Healthy Soybean',
#     'Squash___Powdery_mildew': 'Squash Powdery Mildew',
#     'Strawberry___Leaf_scorch': 'Strawberry Leaf Scorch',
#     'Strawberry___healthy': 'Healthy Strawberry',
#     'Tomato___Bacterial_spot': 'Tomato Bacterial Spot',
#     'Tomato___Early_blight': 'Tomato Early Blight',
#     'Tomato___Late_blight': 'Tomato Late Blight',
#     'Tomato___Leaf_Mold': 'Tomato Leaf Mold',
#     'Tomato___Septoria_leaf_spot': 'Tomato Septoria Leaf Spot',
#     'Tomato___Spider_mites Two-spotted_spider_mite': 'Tomato Spider Mites (Two-Spotted Spider Mite)',
#     'Tomato___Target_Spot': 'Tomato Target Spot',
#     'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Tomato Yellow Leaf Curl Virus',
#     'Tomato___Tomato_mosaic_virus': 'Tomato Mosaic Virus',
#     'Tomato___healthy': 'Healthy Tomato'
# }

# disease_model_path = 'plant_disease_model.pth'
# disease_model = CNN_NeuralNet(3, len(disease_classes))
# disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
# disease_model.eval()

# def predict(image_path):
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),  
#         transforms.ToTensor(),
#     ])
#     image = Image.open(image_path).convert('RGB')
#     tr = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         output = disease_model(tr)
#         predicted_index = output.argmax(dim=1).item()
#         predicted_class = disease_classes[predicted_index]
#         return predicted_class
    
# def init_models():
#     """
#     Initialize the LLM and embedding model settings.
#     """
#     model_name = "mixedbread-ai/mxbai-embed-large-v1"
#     embed_model = MixedbreadAIEmbedding(
#         'emb_07fb3b000f973e7486e883a7206bed120d45f9cb4c066329',
#         model_name=model_name
#     )
#     llm = Groq(
#         model="llama-3.1-8b-instant",
#         api_key="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"
#     )
#     Settings.llm = llm
#     Settings.embed_model = embed_model

# def initialize_chat_engine():
#     """
#     Initialize the chat engine from document data.
#     """
#     global chat_engine
#     init_models()
#     reader = SimpleDirectoryReader(input_files=['disease.pdf'])
#     docs = reader.load_data()

#     index = VectorStoreIndex.from_documents(docs)
#     chat_engine = index.as_chat_engine(
#         chat_mode="context",  
#         verbose=True,
#        system_prompt = ("You are an agricultural chatbot, capable of having general conversations and offering detailed insights on farming techniques, crops, livestock, sustainability practices.")
#         )
#     return chat_engine

# def format_response(text):
#     """
#     Convert the response text into HTML format using markdown.
#     """
#     html = markdown.markdown(text)
#     return html

# def get_disease_diagnosis(image_path):
#     out = predict(image_path)
#     print(out)
#     return out
  
# def provide_disease_management(disease_name):
#     """
#     Provide disease management information based on the disease name.
#     """
#     prompt = f"Plant disease detected: {disease_name}. Provide management and treatment suggestions. If the name of the detected disease has 'healthy' in the name do not provide treatment and suggestions"
#     engine_out = chat_engine.chat(prompt)
    
#     return engine_out

# init_models()
# ce = initialize_chat_engine()
# disease = get_disease_diagnosis(r"C:\Users\SiD\Major Project\apple_scab.jpeg")
# out = provide_disease_management(disease)
# print(type(out.response))
# print(type(out))
# print(dir(out))

import markdown

text = """Management Suggestions:

Early blight is a common fungal disease that affects potatoes, caused by the fungus Alternaria solani. Here are some management and treatment suggestions to help control the disease: **Prevention:** 1. **Crop rotation**: Rotate potatoes with other crops to break the disease cycle. 2. **Remove volunteer potatoes**: Remove any volunteer potatoes that may be growing in the field, as they can serve as a source of infection. 3. **Improve air circulation**: Ensure good air circulation around the plants to reduce moisture and prevent fungal growth. 4. **Water management**: Avoid overhead watering, which can splash spores onto the plants. Instead, use drip irrigation or soaker hoses. **Treatment:** 1. **Fungicides**: Apply fungicides at the first sign of infection, typically in the spring when the plants are in bloom. Copper-based fungicides, such as copper oxychloride, are effective against early blight. 2. **Strobilurin fungicides**: Strobilurin fungicides, such as azoxystrobin, are also effective against early blight. 3. **Organic options**: For organic growers, sulfur-based fungicides or bicarbonate-based products can be used to control early blight. **Cultural controls:** 1. **Resistant varieties**: Plant resistant potato varieties, such as 'Russet Burbank' or 'Shepody', which are less susceptible to early blight. 2. **Sanitation**: Remove any infected tubers or debris from the field to prevent the spread of the disease. 3. **Avoid excessive nitrogen**: Avoid excessive nitrogen fertilization, which can promote lush growth and increase the risk of disease. **Biological controls:** 1. **Beneficial fungi**: Introduce beneficial fungi, such as Trichoderma, which can compete with the early blight fungus for resources. 2. **Predatory insects**: Encourage predatory insects, such as lady beetles or lacewings, which can feed on the early blight fungus. Remember to always follow the label instructions and take necessary precautions when applying fungicides."""
html = markdown.markdown(text)

print(html)