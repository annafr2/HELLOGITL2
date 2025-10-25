import google.generativeai as genai
import os
from gmail_tools import get_labels, search_and_save_emails

# Paste your API KEY here!
API_KEY = "AIzaSyCEOYGpO9U-FJlJAxD36sHTKrGc_pPVRBs"
genai.configure(api_key=API_KEY)

# Define the Functions
tools = [
    {
        "function_declarations": [
            {
                "name": "get_labels",
                "description": "Returns list of all available Gmail labels/tags",
                "parameters": {"type": "object", "properties": {}}
            },
            {
                "name": "search_and_save_emails",
                "description": "Search emails by label and save them to Excel file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "label_name": {
                            "type": "string",
                            "description": "The label name to search for (e.g., LinkedIn, Work, Important)"
                        }
                    },
                    "required": ["label_name"]
                }
            }
        ]
    }
]

model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-exp',
    tools=tools
)

function_map = {
    "get_labels": get_labels,
    "search_and_save_emails": search_and_save_emails
}

def chat():
    """Interactive chat loop"""
    chat_session = model.start_chat()
    
    print("🤖 Gmail Agent Ready!")
    print("\nExample commands:")
    print("  - Show me all my labels")
    print("  - Search for emails with LinkedIn label")
    print("  - Save emails from Work label to file")
    print("\nType 'exit' to quit\n")
    print("=" * 60)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye! 👋")
            break
        
        if not user_input:
            continue
        
        try:
            response = chat_session.send_message(user_input)
            
            while response.candidates[0].content.parts:
                part = response.candidates[0].content.parts[0]
                
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                    function_name = function_call.name
                    function_args = dict(function_call.args)
                    
                    print(f"\n🔧 Executing: {function_name}")
                    
                    function_response = function_map[function_name](**function_args)
                    
                    response = chat_session.send_message(
                        genai.protos.Content(
                            parts=[genai.protos.Part(
                                function_response=genai.protos.FunctionResponse(
                                    name=function_name,
                                    response={'result': function_response}
                                )
                            )]
                        )
                    )
                else:
                    print(f"\n🤖 Agent: {part.text}")
                    break
        except Exception as e:
            print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    chat()