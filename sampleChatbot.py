#importing necessary modules

import speech_recognition as sr
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
from gtts import gTTS
import os
import PyPDF2

def answer_question(question, context):
    # Load pre-trained BERT model and tokenizer fine-tuned on SQuAD
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')

    # Tokenize and encode input text
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract start and end scores from the model output
    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the tokens with the highest start and end scores
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Get the answer span
    answer_tokens = inputs["input_ids"][0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer

def extract_text_from_pdf(pdf_path):
    #extracting text from the pdf file
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        # Iterate through all pages
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

    return text

def recognize_speech():
    # Initialize the recognizer
    recognizer = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source, timeout=5)  # Listen for up to 5 seconds

    try:
        print("Recognizing...")
        # Recognize speech using Google Web Speech API
        text = recognizer.recognize_google(audio)
        print("You said: {}".format(text))
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print("Error with the speech recognition service; {0}".format(e))
        return None
    


if __name__ == "__main__":
    pdf_path = 'questions.pdf'
    #text = extract_text_from_pdf(pdf_path)
    context = """
    Pune Institute of Computer Technology (PICT) has a central library. There are various reasons available in the library such as E-Books, E-Journals, video lectures, CDs/DVDs, project reports. Students can virtue 3 books at a time for a period of 15 days. PICT library books can also be searched remotely on web-OPAC at http://pictlibrary.ourlib.in. A required book can be borrowed after scanning the I-Card. First FIVE Toppers can borrow 5 additional books for a semester. the Book-Bank facility is available in the library on a first come first serve basis. A book Bank facility is also available on a first come first serve basis, students can borrow 3 books for the whole semester. If a book is lost, acopy of the same book (same edition/latest edition) should be replaced to the library. Wi-Fi connectivity is also available to those students who have registered their names for the said purpose. The library also has a separate section "MANTHAN-The Change Maker" is also available for non-technical books. The library also has a separate section Group Study Area. The timing for book issue is from 8.30 a.m. to 7.30 p.m. The timing for the reading hall is 6.00 a.m. to 12.00 a.m. The contents of the figures library are : E-Books, E-Journals, Video Lectures, CDs/DVDs , Project Report. Student can access e-reaource on //10.10.15.220:8080/digital-library."""
    
    while True:
        recognized_text = recognize_speech()
    
        if recognized_text:
            # Now you can use the recognized text as needed in your program
            question = recognized_text
            answer = answer_question(question, context)
            print("Answer : ", answer)
            if answer:
                tts = gTTS(text=answer, lang='en')
                tts.save("output.mp3")
                os.system("start output.mp3")
            else:
                print("Cannot find answer for your question")
        else:
            print("Speech is not recognized")
        

