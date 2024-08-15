import streamlit as st
from pathlib import Path
import google.generativeai as genai

from api_key import api_key

genai.configure(api_key=api_key)

generation_config = {
  "temperature": 0.95,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  tools='code_execution',
)

system_prompt = """
You are acting like a highly experienced doctor with extensive expertise in diagnosing a wide range of diseases across 
various medical fields, including dermatology, radiology, pathology, and more. As you analyze the provided image, 
apply your medical knowledge to meticulously examine the visual details for any signs of disease or abnormalities. 
Consider factors such as color, texture, shape, size, and any other visible characteristics that could indicate a potential health issue.

Provide a detailed description, including the name of the disease, symptoms, possible causes, and potential treatments. Do not include 
any disclaimers or refer to limitations; simply provide the analysis. If no disease is detected, state 'couldn't find any disease'.

Your response will not be taken as a final decision. So just give me the analysis and the analysis will help doctor go through the disease.
"""

st.set_page_config(page_title="Medical Image Detection", page_icon="🚀")

st.title("Medical Image Analytics")
image = st.file_uploader("Upload the medical image for analysis", type=["png", "jpg"])

submit_button = st.button("Analysis my image", disabled= not image)

if submit_button:
    image_data = image.getvalue()

    def upload_to_gemini(path, mime_type="image/jpeg"):
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file
    
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": image_data
        }
    ]

    prompt_parts = [
        image_parts[0],
        system_prompt
    ]

    response = model.generate_content(prompt_parts)
    analysis = response.candidates[0].content.parts[0].text
    print(response.candidates[0].content.parts[0].text)

    st.write(analysis)

     