import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout, add
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from PIL import Image

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet')
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg_model.predict(img, verbose=0)
    return features

# Example image
img_path = 'example.jpg'

# Extract features
features = extract_features(img_path)

# Example data (in a real scenario, you would have a dataset of images and captions)
captions = ["a cat sitting on a sofa", "a cat on a couch", "a feline on a sofa"]
captions_dict = {img_path: captions}

# Preprocess text data
def preprocess_text(captions_dict):
    tokenizer = Tokenizer()
    all_captions = []
    for key in captions_dict:
        for cap in captions_dict[key]:
            all_captions.append('startseq ' + cap + ' endseq')
    
    tokenizer.fit_on_texts(all_captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(c.split()) for c in all_captions)
    
    return tokenizer, vocab_size, max_length

tokenizer, vocab_size, max_length = preprocess_text(captions_dict)

# Create the model
def create_model(vocab_size, max_length):
    # Feature extractor (encoder)
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Sequence processor (decoder)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    # Decoder (merging both models)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = create_model(vocab_size, max_length)

# Define a function to generate captions
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word[yhat]
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

# Example of generating a caption
photo = features.reshape((1, 4096))
caption = generate_caption(model, tokenizer, photo, max_length)
print("Generated Caption:", caption)

# Visualize the image
img = Image.open(img_path)
plt.imshow(img)
plt.axis('off')
plt.show()
