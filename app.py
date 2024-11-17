import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set custom web page title and icon (icon cannot be an image)
st.set_page_config(page_title="Picturify.AI", page_icon="logo.png", layout="wide")

st.markdown(
    """
    <style>
        /* Hide the clip option for text */
        h1 {
            user-select: none; /* Prevents text selection */
        }

        

        /* Make tabs use full width */
        .stTabs {
            width: 100%;
        }

        /* Center align tab names with padding on left and right */
        .stTabs [data-baseweb="tab"] {
            text-align: center; /* Center align text */
            padding: 0 190px; /* Padding on left and right */
            font-size: 50px; /* Increased font size */
        }

        /* Adjust top margin */
        .stApp {
            margin-top: 0 !important; /* Remove top margin */
            padding-top: 0 !important; /* Remove top padding */
        }

        /* Hide footer */
        footer { 
            visibility: hidden; 
        }
        
    </style>
    """,
    unsafe_allow_html=True
)



# Display the logo and website name at the top in the same row
col1, col2 = st.columns([1, 8])
with col1:
    st.image("logo.png", width=120)  # Adjust the width as needed

with col2:
    st.markdown(
        "<h1 style='text-align: left; color: white; font-size: 4em;'>Picturify.AI</h1>",
        unsafe_allow_html=True
    )

# Create main tabs at the top of the page
tabs = st.tabs(["Home", "Caption Generator", "Working"])

# Home Tab
with tabs[0]:
    # Main Title
    st.title("Welcome to Picturify.AI!")
    st.write("Harness the power of deep learning to automatically generate meaningful captions for your images.")

    # Overview of Project
    st.subheader("Overview of Picturify.AI")
    st.write("""
        Picturify.AI is designed to convert visual content into descriptive language. By combining 
        a Convolutional Neural Network (CNN) for extracting image features and a Long Short-Term Memory (LSTM) 
        model with an attention mechanism for language generation, this application is capable of generating human-like captions for images.
        """)
    
    # Key Features Section
    st.subheader("Key Features")
    st.write("""
        - **Real-time Image Captioning**: Generates captions within seconds of image upload.
        - **Attention Mechanism**: The attention layer helps focus on significant areas in an image, resulting in more accurate captions.
        - **User-Friendly Interface**: Simple and intuitive, allowing users to interact without any technical expertise.
        - **Scalability**: Designed to handle multiple users simultaneously and manage high request volumes efficiently.
    """)

    # Sample Use Cases
    st.subheader("Applications")
    st.write("""
        - **Accessibility**: Helps visually impaired users understand images by providing descriptive captions.
        - **E-commerce**: Automatically generates product descriptions for online listings.
        - **Social Media**: Provides content creators with descriptive captions for image posts.
    """)

    # Display project sample image
    st.image("User Interface.png", caption="Example of image captioning with attention mechanism", use_column_width=True)

    # Future Scope and Enhancements
    st.subheader("Future Scope and Enhancements")
    st.write("""
        This project lays the groundwork for numerous potential improvements:
        - **Mobile Integration**: Expand functionality to mobile devices.
        - **Video Captioning**: Adapt the model to generate descriptions for video content.
        - **Multimodal Features**: Incorporate text, audio, or contextual data for richer captions.
    """)


# Working Tab
with tabs[2]:
    st.title("How Picturify.AI Works")
    st.write("Explore the technical workflow that powers our image captioning model.")

    # System Overview
    st.subheader("System Overview")
    st.write("""
        The Picturify.AI system combines CNN for visual feature extraction and LSTM for language modeling, 
        enhanced by an attention mechanism to generate detailed captions. This design enables the model to 
        focus on different areas of the image at each step, enhancing caption relevance and accuracy.
    """)

    # Step-by-Step Workflow
    st.subheader("Workflow Steps")
    
    st.markdown("""
        #### Step 1: Image Upload
        - Users upload an image in formats such as JPEG or PNG through the Streamlit interface.
        
        #### Step 2: Feature Extraction with CNN (ResNet)
        - The CNN (ResNet-50) extracts a 2048-dimensional feature vector representing important visual elements of the image.
        
        #### Step 3: Sequence Generation with LSTM and Attention
        - The extracted features are input to an LSTM with attention. The attention mechanism allows the model to focus on 
          different parts of the image as each word in the caption is generated.
        
        #### Step 4: Caption Display
        - The generated caption is returned to the frontend and displayed alongside the uploaded image for the user to view.
    """)
    
    # Model Diagram
    st.subheader("Technical Architecture")
    st.write("""
        Our architecture uses ResNet-50 for feature extraction and an LSTM network for caption generation, 
        enhanced with an attention mechanism to provide contextually rich captions.
    """)
    st.image("System Flow Chart.png", caption="System Flowchart of CNN-LSTM with Attention Mechanism", use_column_width=True)
    
    # Attention Mechanism Explanation
    st.subheader("Attention Mechanism")
    st.write("""
        Attention is integral to the model, allowing it to focus selectively on parts of the image during caption generation. 
        This ensures captions reflect the most important elements, improving relevance and descriptive accuracy.
    """)
    st.image("Example of Working of Attention Mechanism.png", caption="Visualization of Attention Focus in Caption Generation", use_column_width=True)

    # Technical Specifications
    st.subheader("Technical Specifications")
    st.markdown("""
        - **Feature Extraction**: ResNet-50 CNN model pre-trained on ImageNet
        - **Language Generation**: LSTM network with 512 hidden units
        - **Attention Mechanism**: Highlights significant regions in the image for each generated word
        - **Deployment**: Cloud-hosted backend for real-time caption generation, integrated with Streamlit frontend
    """)

    # Backend Processing Diagram (Optional)
    st.subheader("Backend Processing Flow")
    st.write("The backend flow from image upload to caption generation is illustrated below:")
    st.image("Diagram of the CNN-LSTM Architecture.png", caption="Backend Processing Flow Diagram", use_column_width=True)
    
    # Conclusion
    st.subheader("Conclusion")
    st.write("""
        This workflow demonstrates how modern AI techniques in computer vision and natural language processing can create 
        applications that interpret visual content into meaningful text. With further enhancements, Picturify.AI can continue 
        to serve diverse needs across various fields.
    """)


# Caption Generator Tab
with tabs[1]:
    # Load ResNet50 model (to match the expected feature shape)
    resnet_model = ResNet50(weights="imagenet")
    resnet_model = Model(inputs=resnet_model.inputs, outputs=resnet_model.layers[-2].output)

    # Load your trained model
    model = tf.keras.models.load_model('mymodel.h5')

    # Load the tokenizer
    with open('tokenizer.pkl', 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)

    # Streamlit app
    st.title("Image Caption Generator")
    st.markdown(
        "Upload an image, and this app will generate a caption for it using a trained Attention Based ResNet model."
    )

    # Upload image
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    # Process uploaded image
    if uploaded_image is not None:
        st.subheader("Uploaded Image")

        # Center the uploaded image and caption
        with st.container():
            col = st.columns(1)  # Create one column for centering
            with col[0]:
                
                st.image(uploaded_image, caption="Uploaded Image")  # Display the image with a caption

        st.subheader("Generated Caption")
        # Display loading spinner while processing
        with st.spinner("Generating caption..."):
            # Load image
            image = load_img(uploaded_image, target_size=(224, 224))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)

            # Extract features using ResNet50
            image_features = resnet_model.predict(image)

            # Predict caption
            def predict_caption(model, image_features, tokenizer, max_caption_length):
                sequence = [tokenizer.word_index['startseq']]
                for _ in range(max_caption_length):
                    sequence_padded = pad_sequences([sequence], maxlen=max_caption_length)
                    yhat = model.predict([image_features, sequence_padded], verbose=0)
                    yhat = np.argmax(yhat)
                    word = tokenizer.index_word.get(yhat, None)
                    if word is None or word == 'endseq':
                        break
                    sequence.append(yhat)
                return ' '.join([tokenizer.index_word[i] for i in sequence if i not in [tokenizer.word_index['startseq'], tokenizer.word_index['endseq']]])

            # Generate the caption
            max_caption_length = 34  # Replace with your actual max caption length
            generated_caption = predict_caption(model, image_features, tokenizer, max_caption_length)

            # Display the generated caption
            st.write(generated_caption)