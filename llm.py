import openai

openai.api_key = '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'

#setting context to the chatgpt
messages = [
    {"role" : "system", "content" : "You are assistant in the library"}, 
]

while True:
    message = input("User : ")
    if message:
        messages.append(
            {"role": "user", "content" : message}, 
        )

        chat = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo", 
            messages = messages
        )

    reply = chat.choices[0].message.content
    print(f"ChatBot:{reply}")
    messages.append({"role": "assistant", "content" : reply})

