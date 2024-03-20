import streamlit as st
import os
import torch
from PIL import Image
import openai

@st.cache_data()
def load_model():
    # Make sure to point to the correct model weights file
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=r"C:\Users\aravs\PycharmProjects\DMPA\yolov5\runs\train\exp3\weights\last.pt",
                           force_reload=True)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


@st.cache_resource(show_spinner=False)
def get_openai_response(prompt, api_key):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=50,
        api_key=api_key
    )
    return response.choices[0].text.strip()


def process_image(image_path, user_name, model, api_key, action):
    global labels
    global inventory_dict
    results = model(image_path)
    preds = results.pred[0]
    dictionary = {}

    for pred in preds:
        label = labels[int(pred[-1])]
        confidence = pred[4].item()
        if confidence > 0.25:  # Confidence threshold
            dictionary[label] = dictionary.get(label, 0) + 1

    sentence = ", ".join(f"{count} {label}{'s' if count > 1 else ''}" for label, count in dictionary.items())

    print("Changes made in the following itemsets: \n")
    if action=="Borrowed":
        for label, count in dictionary.items():
            if dictionary[label] >= count:
                inventory_dict[label] -= count

    elif action=="Returned":
        for label, count in dictionary.items():
            if dictionary[label] + count <= 100:
                inventory_dict[label] += count


    for label, count in inventory_dict.items():
        if inventory_dict[label] != 100:
            print(f"{label} :: {count}")

    print("\n\n\n")

    if sentence:
        prompt = f"I have listed which item and how many of them have been {action} to/from the electronics inventory- {sentence}. The listed items have been {action} by {user_name}. Form a sentence from all the data."
        generated_sentence = get_openai_response(prompt, api_key)
        return generated_sentence
    else:
        return "No items detected."


st.title('Circuit Elements Inventory Manager')

# Load model
model = load_model()

# Retrieve API key from Streamlit secrets
api_key = st.secrets["openai_api_key"]

labels = ['Button', 'Buzzer', 'Capacitor Jumper', 'Capacitor Network', 'Capacitor', 'Clock', 'Connector', 'Diode',
          'EM',
          'Electrolytic Capacitor', 'Electrolytic capacitor', 'Ferrite Bead', 'Flex Cable', 'Fuse', 'IC',
          'Inductor',
          'Jumper', 'Led', 'Pads', 'Pins', 'Potentiometer', 'RP', 'Resistor Jumper', 'Resistor Network', 'Resistor',
          'Switch', 'Test Point', 'Transducer', 'Transformer', 'Transistor', 'Unknown Unlabeled']

user_name = st.text_input("Enter User Name", "")
action = st.text_input("Are you borrowing or returning")

inventory_dict = {label: 100 for label in labels}

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_image is not None:
    # Display the image
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # When the button is pressed
    if st.button('Update Inventory'):
        with st.spinner('Generating sentence...'):
            # Save the uploaded image to a temporary file
            temp_path = f"temp_image.{uploaded_image.name.split('.')[-1]}"
            with open(temp_path, "wb") as file:
                file.write(uploaded_image.getbuffer())

            # Generate the sentence
            generated_sentence = process_image(temp_path, user_name, model, api_key, action)

            # Display the sentence
            if generated_sentence:
                st.success('Updated Inventory!')
                st.write(generated_sentence)
            else:
                st.error("No items detected or unable to generate sentence.")

            # Clean up the temp image file
            os.remove(temp_path)
