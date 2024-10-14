import pandas as pd

# Function to load symptom-to-medicine data from Excel
def load_symptom_data(file_path):
    try:
        # Load the Excel file
        df = pd.read_excel(file_path)
        
        # Ensure the first two columns are 'Symptom' and 'Medicine'
        df.columns = ['Symptom', 'Medicine']  # Rename columns if necessary
        
        return df
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None

# Function to recommend medicine based on symptom from the Excel data
def recommend_medicine(symptom, df):
    symptom = symptom.lower()
    # Look up the symptom in the DataFrame
    row = df[df['Symptom'].str.lower() == symptom]
    if not row.empty:
        return row['Medicine'].values[0]
    else:
        return 'Consult a doctor'

# Main function to test loading data and getting a recommendation
def main():
    # Load the symptom data from the Excel file
    file_path = 'symptoms.xlsx'
    df = load_symptom_data(file_path)
    
    if df is not None:
        # Get user input for the symptom
        symptom = input("Enter your symptom: ")
        
        # Get the recommended medicine
        medicine = recommend_medicine(symptom, df)
        print(f"Recommended Medicine: {medicine}")

if __name__ == '__main__':
    main()
