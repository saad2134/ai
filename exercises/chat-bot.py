import random
import re
import json
from datetime import datetime

class SimpleChatBot:
    def __init__(self):
        self.name = "AI Assistant"
        self.responses = {
            'greeting': [
                "Hello! How can I help you today?",
                "Hi there! What can I do for you?",
                "Greetings! How may I assist you?"
            ],
            'farewell': [
                "Goodbye! Have a great day!",
                "See you later! Take care!",
                "Bye! Feel free to chat again!"
            ],
            'thanks': [
                "You're welcome!",
                "Happy to help!",
                "Anytime! Is there anything else?"
            ],
            'name': [
                f"I'm {self.name}, your friendly AI assistant!",
                f"You can call me {self.name}.",
                f"I'm {self.name}. How can I assist you today?"
            ],
            'time': [
                f"The current time is {datetime.now().strftime('%H:%M')}",
                f"It's {datetime.now().strftime('%I:%M %p')} now"
            ],
            'date': [
                f"Today is {datetime.now().strftime('%A, %B %d, %Y')}",
                f"The date is {datetime.now().strftime('%m/%d/%Y')}"
            ],
            'weather': [
                "I don't have access to real-time weather data, but you can check your weather app!",
                "For accurate weather information, please check a reliable weather service."
            ],
            'joke': [
                "Why don't scientists trust atoms? Because they make up everything!",
                "Why did the scarecrow win an award? He was outstanding in his field!",
                "Why don't eggs tell jokes? They'd crack each other up!"
            ],
            'default': [
                "I'm not sure I understand. Could you rephrase that?",
                "That's interesting! Can you tell me more?",
                "I see. What would you like to know about that?",
                "Could you elaborate on that?"
            ]
        }
        
        self.patterns = {
            r'hello|hi|hey|greetings': 'greeting',
            r'goodbye|bye|see you|later': 'farewell',
            r'thank you|thanks|appreciate': 'thanks',
            r'what is your name|who are you': 'name',
            r'time|what time is it': 'time',
            r'date|what day is it|today': 'date',
            r'weather|temperature|forecast': 'weather',
            r'tell me a joke|make me laugh': 'joke',
            r'how are you|how do you do': 'how_are_you'
        }
    
    def get_response(self, user_input):
        user_input = user_input.lower().strip()
        
        # Check for specific patterns
        for pattern, category in self.patterns.items():
            if re.search(pattern, user_input):
                if category == 'how_are_you':
                    return "I'm doing great, thank you! How are you?"
                return random.choice(self.responses[category])
        
        # Check for questions
        if user_input.endswith('?'):
            if 'your name' in user_input:
                return random.choice(self.responses['name'])
            elif 'time' in user_input:
                return random.choice(self.responses['time'])
            elif 'date' in user_input:
                return random.choice(self.responses['date'])
            elif 'weather' in user_input:
                return random.choice(self.responses['weather'])
            else:
                return "That's an interesting question! I'm still learning though."
        
        # Default response
        return random.choice(self.responses['default'])
    
    def chat(self):
        print(f"=== {self.name} ===")
        print("Type 'quit' to exit the chat")
        print("Hello! I'm your AI assistant. How can I help you today?\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print(f"{self.name}: {random.choice(self.responses['farewell'])}")
                    break
                elif user_input == '':
                    continue
                
                response = self.get_response(user_input)
                print(f"{self.name}: {response}\n")
                
            except KeyboardInterrupt:
                print(f"\n{self.name}: {random.choice(self.responses['farewell'])}")
                break
            except Exception as e:
                print(f"{self.name}: Sorry, I encountered an error. Could you try again?")
                print(f"Error: {e}")

class AdvancedChatBot(SimpleChatBot):
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.user_name = None
    
    def remember_user_info(self, user_input):
        # Simple name extraction
        name_pattern = r'my name is (\w+)'
        match = re.search(name_pattern, user_input.lower())
        if match and not self.user_name:
            self.user_name = match.group(1).title()
            return f"Nice to meet you, {self.user_name}!"
        
        # Remember preferences
        if 'like' in user_input and 'don\'t like' not in user_input:
            if 'music' in user_input:
                return "I'll remember that you like music!"
            elif 'movies' in user_input:
                return "Great to know you enjoy movies!"
        
        return None
    
    def get_response(self, user_input):
        # Store conversation
        self.conversation_history.append(f"You: {user_input}")
        
        # Try to remember user info
        memory_response = self.remember_user_info(user_input)
        if memory_response:
            self.conversation_history.append(f"Bot: {memory_response}")
            return memory_response
        
        # Personalized responses
        if self.user_name:
            personalized_responses = {
                'greeting': [f"Hello {self.user_name}! How can I help you?"],
                'default': [f"That's interesting, {self.user_name}. Can you tell me more?"]
            }
            
            for pattern, category in self.patterns.items():
                if re.search(pattern, user_input.lower()):
                    if category in personalized_responses:
                        response = random.choice(personalized_responses[category])
                        self.conversation_history.append(f"Bot: {response}")
                        return response
        
        # Get base response
        response = super().get_response(user_input)
        self.conversation_history.append(f"Bot: {response}")
        return response

def demo_chat_bot():
    print("Choose chatbot type:")
    print("1. Simple ChatBot")
    print("2. Advanced ChatBot (with memory)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        bot = SimpleChatBot()
    else:
        bot = AdvancedChatBot()
    
    bot.chat()

if __name__ == "__main__":
    demo_chat_bot()