from gtts import gTTS
import os

# Specify the text you want to convert
text = "Hello, I am Swarup"

# Create a gTTS object
tts = gTTS(text=text, lang='en')

# Save the generated speech to a file (optional)
tts.save("output.mp3")

# Play the generated speech (requires a media player like VLC)
os.system("start output.mp3")
