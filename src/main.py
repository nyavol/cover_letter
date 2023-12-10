import torch
from io import BytesIO
import PyPDF2
import os

from fastapi import FastAPI, Response, HTTPException
from fastapi import Request, File
from urllib.parse import urlsplit
import requests
from bs4 import BeautifulSoup

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from keybert import KeyBERT
from openai import OpenAI

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI()

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)
kw_model = KeyBERT(model='all-mpnet-base-v2')

model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
model.eval()


def is_pdf(content_type):
    return content_type.lower() == 'application/pdf'


def is_url(url):
    try:
        result = urlsplit(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def read_pdf(file_contents):
    try:
        pdf_file = BytesIO(file_contents)
        pdf_reader = PyPDF2.PdfFileReader(pdf_file)
        text = ""
        for page_number in range(pdf_reader.numPages):
            page = pdf_reader.getPage(page_number)
            text += page.extract_text()

        return text

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading PDF: {e}")


def extract_keywords(generated_text, num_keywords=5):
    keywords = kw_model.extract_keywords(generated_text,
                                         keyphrase_ngram_range=(1, 2),
                                         stop_words='english',
                                         highlight=False,
                                         top_n=num_keywords)

    keywords_list = list(dict(keywords).keys())

    return keywords_list


def generate_cover_letter(job_description, resume, length=500):
    resume_keywords = extract_keywords(resume, num_keywords=5)
    job_keywords = extract_keywords(job_description, num_keywords=5)

    prompt = f"Dear Recruiter,\n\nI am writing to express my strong interest in the [Job Title] position at [Company Name]." \
             f"I have a strong knowledge of {', '.join(resume_keywords)}. Therefore, I am confident in my ability to contribute to the company's mission."
    full_prompt = f"{prompt} Moreover, my expertise in {', '.join(job_keywords)}, can be of huge help to succeed in the position."
    prompt_tokens = tokenizer.encode(full_prompt, return_tensors='pt').to(device)

    cover_letter = ""
    while len(cover_letter) < length:
        output = model.generate(prompt_tokens, max_length=length+100,
                                num_beams=5, temperature=0.8,
                                no_repeat_ngram_size=2,
                                top_k=50, top_p=0.92,
                                do_sample=True,
                                pad_token_id=tokenizer.eos_token_id,
                                attention_mask=torch.ones_like(prompt_tokens))

        cover_letter = tokenizer.decode(output[0], skip_special_tokens=True)

    return cover_letter


def generate_cover_letter_gpt3(job_description, resume):
    prompt = f"My resume is {resume} and I am applying to {job_description}."
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Generate a personalized cover letter based on the following information."},
            {"role": "user", "content": prompt},
        ],
    )
    cover_letter = response.choices[0].text.lstrip()
    return cover_letter

@app.get("/api")
async def main():
    return {"success": True}

@app.post("/api/generate_cover")
async def gen_cover(request: Request):
    form = await request.form()
    try:
        resume = form["resume_text"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No resume attached: {e}")

    try:
        posting = form["job_posting"]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No job description attached: {e}")

    try:
        is_pdf(resume.content_type)
        resume = await form["resume_text"].read()
        resume = read_pdf(resume)
    except Exception as e:
        pass

    if is_url(posting):
        try:
            posting = BeautifulSoup(requests.get(posting).text)
            metas = posting.find_all('meta')
            posting = [meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description' ]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"URL is Incorrect: {e}")

    length = form["cover_length"]

    try:
        length = int(length)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Wrong type of cover letter length: {e}")

    use_gpt3 = form["use_gpt3"]
    if use_gpt3:
        try:
            cover_letter = generate_cover_letter_gpt3(posting, resume)
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Error from OpenAI: {e}")
    else:
        cover_letter = generate_cover_letter(posting, resume, length)

    return cover_letter

