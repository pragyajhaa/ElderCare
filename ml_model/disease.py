import cv2
import numpy as np

# Sample dataset with common symptoms and recommended medicines
symptom_to_medicine = {
    'headache': 'Paracetamol',
    'fever': 'Ibuprofen',
    'cough': 'Cough Syrup',
    'knee pain': 'Anti-inflammatory cream',
    'stomach ache': 'Antacid',
    'cold': 'Decongestant',
    'back pain': 'Pain relief gel',
    'muscle pain': 'Muscle relaxant',
    'sore throat': 'Throat lozenges',
    'allergy': 'Antihistamines',
    'dizziness': 'Motion sickness tablets'
}

# Function to display text on the screen
def display_text(image, text, position, font_scale=1, color=(0, 255, 0), thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# Function to get user input through OpenCV
def get_user_input(prompt):
    # Create a blank image
    image = np.zeros((500, 800, 3), dtype=np.uint8)
    display_text(image, prompt, (20, 250), font_scale=1, color=(255, 255, 255))
    
    cv2.imshow('Input', image)
    
    # Simulate user input (you would typically use a GUI library for better text input)
    user_input = input(prompt + "\n> ")
    
    cv2.destroyAllWindows()
    
    return user_input

# Function to recommend medicine based on symptom
def recommend_medicine(symptom):
    # Use the dataset to find the recommended medicine
    symptom = symptom.lower()
    medicine = symptom_to_medicine.get(symptom, 'Consult a doctor')
    return medicine

# Function to simulate booking an online consultation
def book_online_consultation(age, gender):
    # This would typically involve a real API call to a scheduling system
    print(f"Booking online consultation for a {age}-year-old {gender}.")
    print("Consultation booked successfully!")

# Main function to handle the entire process
def main():
    print("Welcome to the Health Assistant!")
    
    # Get user inputs
    symptom = get_user_input("Please enter your symptom:")
    age = get_user_input("Please enter your age:")
    gender = get_user_input("Please enter your gender:")
    
    # Recommend medicine
    medicine = recommend_medicine(symptom)
    print(f"Recommended Medicine: {medicine}")
    
    # Book an online consultation
    book_online_consultation(age, gender)
    
if __name__ == '__main__':
    main()
