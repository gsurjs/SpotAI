# Spot: Cognitive Companion

![alt text](https://ibb.co/FLBfXgCr)

Spot: Cognitive Companion is an AI-powered robotic assistant designed to enhance human-robot interactions by integrating dynamic AI capabilities, user recognition, and environmental awareness. This project leverages facial recognition, large language models, and vision language models to create a more interactive and intelligent robotic assistant.

## Features
- **Human-Robot Interaction**: Uses AI to improve engagement with users.
- **Large Language Model (LLM) & Vision Language Model (VLM)**:
  - Supports OpenAI, Google Gemini, and Groq Llama3 for natural language interactions.
  - Processes real-world visual surroundings for contextual responses.
- **Speech-to-Text**: Uses `faster-whisper` for real-time speech recognition.
- **Text-to-Speech**: Offline FREE speech synthesis using `pyttsx3`.
- **Object Detection**: Features such as identifying objects and surroundings

## Installation

### Prerequisites
Ensure you have Python installed (>=3.8).

### Steps
1. Clone this repository:
   ```sh
   git clone https://github.com/YOUR_USERNAME/SpotAI.git
   cd SpotAI
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the assistant:
   ```sh
   python assistant.py
   ```

## Usage
- Say "spot" followed by your voice prompt.
- The assistant will process your command and respond.
- It may use clipboard content, capture webcam images, or take screenshots if necessary.

## Acknowledgments
This project was funded by a grant from the **Department of Defense** and developed at **Georgia State University** under the mentorship of **Dr. Anu G Bourgeois**.

## Contributing
Feel free to fork this repository and submit pull requests.

## License
This project is licensed under the MIT License.

