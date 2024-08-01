import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import re
from PIL import Image, ImageDraw
import time
import requests
import numpy as np
from io import BytesIO
import pytesseract  # OCR tool

# Set page configuration
st.set_page_config(page_title="Welcome to NutriDoc APP", layout="wide")

load_dotenv()

# Configure API key for Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    st.error("Google API key not found. Please set the GOOGLE_API_KEY environment variable.")

@st.cache_data
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        return Image.open(BytesIO(response.content)).convert("RGBA")
    else:
        raise Exception("Failed to download image from Google Drive")

def make_circular(image):
    np_image = np.array(image)
    h, w = np_image.shape[:2]

    # Create alpha mask
    alpha = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(alpha)
    draw.pieslice([0, 0, w, h], 0, 360, fill=255)
    np_alpha = np.array(alpha)

    # Add alpha layer to the image
    np_image = np.dstack((np_image[:, :, :3], np_alpha))
    return Image.fromarray(np_image)

def get_gemini_response(input_prompt, retries=3, delay=5):
    model = genai.GenerativeModel("gemini-pro")
    attempt = 0
    while attempt < retries:
        try:
            response = model.generate_content([input_prompt])
            if not response.text:
                if response.safety_ratings:
                    st.warning(f"Response blocked due to safety ratings: {response.safety_ratings}")
                else:
                    st.warning("No valid response text received.")
                raise Exception("No valid response text received.")
            return response.text
        except Exception as e:
            st.error(f"Attempt {attempt + 1} failed: {e}")
            attempt += 1
            time.sleep(delay * attempt)
    raise Exception("All retry attempts failed. Please check the service status or contact support.")

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def parse_nutrition_response(response_text):
    # Define multiple regular expression patterns to extract the nutritional information
    patterns = [
        r'(\w+)\s*-\s*(\d+)\s*calories',
        r'(\w+)\s*:\s*(\d+)\s*calories',
        # Add more patterns if necessary
    ]

    nutrition_data = {}
    for pattern in patterns:
        matches = re.findall(pattern, response_text)
        for match in matches:
            nutrition_data[match[0]] = int(match[1])
    
    return nutrition_data

@st.cache_data
def plot_bar_chart(nutrition_data):
    if not nutrition_data:
        st.warning("No nutritional data found to plot the bar chart.")
        return None

    labels = list(nutrition_data.keys())
    sizes = list(nutrition_data.values())
    
    # Assuming items with less than 100 calories are considered healthy
    colors = ['green' if calories < 100 else 'red' for calories in sizes]

    fig, ax = plt.subplots()
    ax.bar(labels, sizes, color=colors)

    ax.set_ylabel('Calories')
    ax.set_title('Nutritional Content by Food Item')
    
    # Rotate x-axis labels for better readability
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")

    return fig

def calculate_bmi(weight, height_cm):
    bmi = weight / (height_cm / 100) ** 2
    return bmi

def height_in_inches_to_cm(inches):
    return inches * 2.54

def get_diet_advice(bmi, health_issues, nutrition_data):
    # Example function to generate diet advice based on inputs
    advice = []

    # Determine if the user needs specific dietary recommendations based on BMI and health issues
    if bmi < 18.5:
        advice.append("You are underweight. Focus on calorie-dense foods and healthy fats.")
    elif bmi >= 25:
        advice.append("You are overweight. Focus on a balanced diet with portion control.")
    
    if "Heart Issue" in health_issues:
        advice.append("Limit saturated fats and cholesterol. Focus on lean proteins and fiber-rich foods.")
    if "Diabetes" in health_issues:
        advice.append("Monitor carbohydrate intake. Choose whole grains and avoid added sugars.")
    if "Hypertension" in health_issues:
        advice.append("Limit sodium intake. Focus on fruits, vegetables, and low-fat dairy products.")
    if "PCOD" in health_issues:
        advice.append("Focus on whole foods and avoid processed foods. Balance carbohydrates with proteins.")

    # Analyze nutrition data for specific food recommendations
    if nutrition_data:
        for item, calories in nutrition_data.items():
            if calories > 200:
                advice.append(f"Avoid excessive calorie intake from {item}.")
    
    return advice

# Download and display the logo
logo_url = "https://drive.google.com/uc?id=11iHX_WYVXY8Dz2QtJctPzuwgMwwnf2b1"

@st.cache_data
def get_logo_image(url):
    try:
        logo_image = download_image(url)
        logo_image = make_circular(logo_image)
        logo_image.save("/tmp/logo_circle.png")
        return "/tmp/logo_circle.png"
    except Exception as e:
        st.warning(f"Logo image not found: {e}")
        return None

logo_image_path = get_logo_image(logo_url)
if logo_image_path:
    st.image(logo_image_path, width=200)

st.markdown("# Welcome to NutriDoc APP", unsafe_allow_html=True)
st.markdown("### Enter your profile information")
sex = st.radio("Sex", ("Male", "Female"))
health_issues = st.multiselect("Health Issues", ["Heart Issue", "Diabetes", "Hypertension", "PCOD", "None"])
dietary_preference = st.radio("Dietary Preference", ("Vegetarian", "Non-Vegetarian", "Vegan"))

st.markdown("### Enter your details to calculate BMI")
col1, col2 = st.columns(2)
with col1:
    inches = st.number_input("Height (in inches)", min_value=0.0, format="%.2f")
with col2:
    weight = st.number_input("Weight (in kg)", min_value=0.0, format="%.2f")

if inches > 0 and weight > 0:
    height_cm = height_in_inches_to_cm(inches)
    bmi = calculate_bmi(weight, height_cm)
    st.markdown(f"### Your BMI: {bmi:.2f}")
    
    input_prompt = f"""
    You are an expert in nutrition where you need to see the food items from the image
    and calculate the total calories. Also provide the details of every food item with calories intake
    in the below format:

    1. Item - no of calories
    2. Item - no of calories
    ----
    ----
    Finally, you can also mention whether the food is healthy or not and also mention the percentage split of the ratio of carbohydrates, fats, fibers, sugars, and other important things required in our diet.
    Also generate a bar chart for all the food ingredients like how much carbohydrates etc. with different colors.

    Strictly, consider the following details:
    - Sex: {sex}
    - Health Issues: {", ".join(health_issues)}
    - Dietary Preference: {dietary_preference}
    - Height: {height_cm:.2f} cm
    - Weight: {weight} kg
    - BMI: {bmi:.2f}

    Based on the above details, also provide insights on how the food in the image affects the person's health.
    """

    quote_prompt = f"""
    Generate a humorous quote related to the food items in the image, considering the context of the user's profile:
    - Sex: {sex}
    - Health Issues: {", ".join(health_issues)}
    - Dietary Preference: {dietary_preference}
    - BMI: {bmi:.2f}
    """

    replace_food_prompt = f"""
    Generate a table of food items from the image that can be replaced with healthier alternatives:
    - Sex: {sex}
    - Health Issues: {", ".join(health_issues)}
    - Dietary Preference: {dietary_preference}
    - BMI: {bmi:.2f}
    """

    healthier_alternatives_prompt = f"""
    Generate a table of healthier alternatives for the food items in the image:
    - Sex: {sex}
    - Health Issues: {", ".join(health_issues)}
    - Dietary Preference: {dietary_preference}
    - BMI: {bmi:.2f}
    """

    uploaded_file = st.file_uploader("Upload an image of your food", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Extract text from image
        extracted_text = extract_text_from_image(image)
        
        # Combine input prompt with extracted text
        combined_prompt = f"{input_prompt}\n\nExtracted text from image:\n{extracted_text}"
        
        # Get response from Gemini Pro
        try:
            response_text = get_gemini_response(combined_prompt)
            st.markdown(f"## Nutrition Analysis")
            st.markdown(response_text)
            
            # Parse nutrition data from response
            nutrition_data = parse_nutrition_response(response_text)
            
            # Plot bar chart
            bar_chart = plot_bar_chart(nutrition_data)
            if bar_chart:
                st.pyplot(bar_chart)
            
            # Get diet advice
            diet_advice = get_diet_advice(bmi, health_issues, nutrition_data)
            st.markdown(f"## Diet Advice")
            for advice in diet_advice:
                st.markdown(f"- {advice}")
            
            # Get humorous quote
            humorous_quote = get_gemini_response(quote_prompt)
            st.markdown(f"## Humorous Quote")
            st.markdown(humorous_quote)
            
            # Get healthier alternatives
            healthier_alternatives = get_gemini_response(healthier_alternatives_prompt)
            st.markdown(f"## Healthier Alternatives")
            st.markdown(healthier_alternatives)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
