class MedicalExpertSystem:
    def __init__(self):
        self.rules = {
            'flu': {
                'symptoms': ['fever', 'cough', 'headache', 'fatigue'],
                'treatment': 'Rest, fluids, and over-the-counter fever reducers',
                'severity': 'moderate'
            },
            'common_cold': {
                'symptoms': ['runny_nose', 'sneezing', 'sore_throat', 'cough'],
                'treatment': 'Rest, hydration, and over-the-counter cold medicine',
                'severity': 'mild'
            },
            'covid_19': {
                'symptoms': ['fever', 'cough', 'shortness_of_breath', 'loss_of_taste'],
                'treatment': 'Isolate and consult healthcare provider for testing',
                'severity': 'high'
            },
            'allergy': {
                'symptoms': ['sneezing', 'itchy_eyes', 'runny_nose', 'rash'],
                'treatment': 'Antihistamines and avoid allergens',
                'severity': 'mild'
            },
            'strep_throat': {
                'symptoms': ['sore_throat', 'fever', 'swollen_lymph_nodes', 'difficulty_swallowing'],
                'treatment': 'Antibiotics (prescription required)',
                'severity': 'moderate'
            }
        }
        
        self.symptoms_list = [
            'fever', 'cough', 'headache', 'fatigue', 'runny_nose',
            'sneezing', 'sore_throat', 'shortness_of_breath', 'loss_of_taste',
            'itchy_eyes', 'rash', 'swollen_lymph_nodes', 'difficulty_swallowing'
        ]
    
    def diagnose(self, patient_symptoms):
        matches = {}
        
        for condition, data in self.rules.items():
            symptom_match = len(set(patient_symptoms) & set(data['symptoms']))
            total_symptoms = len(data['symptoms'])
            confidence = symptom_match / total_symptoms
            matches[condition] = confidence
        
        # Return diagnosis with highest confidence
        if matches:
            diagnosis = max(matches, key=matches.get)
            confidence = matches[diagnosis]
            return diagnosis, confidence
        return None, 0
    
    def get_treatment(self, condition):
        return self.rules.get(condition, {}).get('treatment', 'Consult a doctor')
    
    def get_severity(self, condition):
        return self.rules.get(condition, {}).get('severity', 'unknown')

def run_medical_expert_system():
    expert = MedicalExpertSystem()
    
    print("=== Medical Diagnosis Expert System ===")
    print("Available symptoms:")
    for i, symptom in enumerate(expert.symptoms_list, 1):
        print(f"{i}. {symptom.replace('_', ' ')}")
    
    print("\nEnter your symptoms (comma-separated numbers, e.g., 1,3,5):")
    
    try:
        symptom_nums = input("Your symptoms: ").split(',')
        patient_symptoms = [expert.symptoms_list[int(num.strip())-1] for num in symptom_nums]
        
        print(f"\nReported symptoms: {[s.replace('_', ' ') for s in patient_symptoms]}")
        
        diagnosis, confidence = expert.diagnose(patient_symptoms)
        
        if diagnosis and confidence > 0.3:
            print(f"\n🔍 DIAGNOSIS: {diagnosis.replace('_', ' ').title()}")
            print(f"📊 Confidence: {confidence:.1%}")
            print(f"⚠️  Severity: {expert.get_severity(diagnosis).upper()}")
            print(f"💊 Recommended Treatment: {expert.get_treatment(diagnosis)}")
            
            if expert.get_severity(diagnosis) == 'high':
                print("\n🚨 URGENT: Please consult a healthcare professional immediately!")
        else:
            print("\n❓ No clear diagnosis. Please consult a doctor for proper evaluation.")
            
    except (ValueError, IndexError):
        print("Invalid input! Please enter numbers separated by commas.")

if __name__ == "__main__":
    run_medical_expert_system()