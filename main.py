import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

app = FastAPI()

# قراءة مفتاح الـ API بأمان من متغيرات البيئة
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

class AnalysisRequest(BaseModel):
    resume_text: str
    job_description: str

@app.get("/")
def home():
    return {"message": "CVFit Backend is Running Live!"}

@app.post("/analyze-cv")
def analyze_cv(data: AnalysisRequest):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""
        أنت خبير في توظيف الكفاءات ومراجعة السير الذاتية.
        قم بمقارنة السيرة الذاتية التالية مع متطلبات الوظيفة الشاغرة وقدم تحليلاً شاملاً.
        
        السيرة الذاتية:
        {data.resume_text}
        
        الوصف الوظيفي:
        {data.job_description}
        
        يرجى تقديم:
        1. نسبة التوافق الإجمالية (كنسبة مئوية).
        2. المهارات والكلمات المفتاحية المفقودة.
        3. 3 نصائح عملية لتحسين السيرة الذاتية لهذه الوظيفة.
        """
        
        response = model.generate_content(prompt)
        return {"result": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
