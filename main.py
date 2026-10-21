import os
import io
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai
from pypdf import PdfReader

# 1. تهيئة تطبيق FastAPI
app = FastAPI(
    title="CVFit AI Master Engine",
    description="محرك الذكاء الاصطناعي الشامل والكامل لخدمات السيرة الذاتية والمقابلات الشخصية",
    version="3.0.0"
)

# 2. إعدادات التواصل (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. ضبط مفتاح Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def get_ai_model():
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY غير مضبوط في متغيرات البيئة")
    return genai.GenerativeModel('gemini-1.5-flash')

# دالة مساعدة لاستخراج النص من ملف PDF
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        pdf_reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        if not text.strip():
            raise Exception("تعذر استخراج نص واضح من ملف الـ PDF.")
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ في قراءة ملف PDF: {str(e)}")

# ---------------------------------------------------------
# نماذج البيانات (Pydantic Models)
# ---------------------------------------------------------
class AnalyzeRequest(BaseModel):
    cv_text: str
    job_description: Optional[str] = "General Evaluation"

class GenerateCVRequest(BaseModel):
    full_name: str
    target_job: str
    experience_summary: str
    skills: str
    education: str

class TailorCVRequest(BaseModel):
    current_cv: str
    target_job_description: str

class InterviewPrepRequest(BaseModel):
    cv_text: str
    job_title: str

class CoverLetterRequest(BaseModel):
    cv_text: str
    job_description: str
    company_name: Optional[str] = "الشركة المستهدفة"

class SimpleCVRequest(BaseModel):
    cv_text: str

class TranslateRequest(BaseModel):
    cv_text: str
    target_language: str = "English"

# ---------------------------------------------------------
# المسارات والخدمات الـ 11 (Endpoints)
# ---------------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "CVFit AI Master Engine is running with all 11 features!",
        "docs": "/docs"
    }

# 1. 📄 قراءة واستخراج النص من ملف PDF
@app.post("/parse-pdf")
async def parse_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text_from_pdf(contents)
    return {"extracted_text": text}

# 2. 🔍 تحليل الـ CV وتحديد نسبة التوافق
@app.post("/analyze-cv")
def analyze_cv(data: AnalyzeRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        قم بدور خبير توظيف وراجع السيرة الذاتية التالية مقارنة بوصف الوظيفة:
        وصف الوظيفة: {data.job_description}
        السيرة الذاتية: {data.cv_text}
        قدم تقريراً باللغة العربية يشمل: نسبة التوافق (من 100%)، نقاط القوة، الثغرات والمهارات الناقصة، ونصائح لتخطي أنظمة ATS.
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 3. 📝 إنشاء CV جديد من الصفر
@app.post("/generate-cv")
def generate_cv(data: GenerateCVRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        اكتب محتوى سيرة ذاتية احترافية ومكتملة متوافقة مع ATS بناءً على:
        الاسم: {data.full_name} | الوظيفة المستهدفة: {data.target_job}
        الخبرات: {data.experience_summary} | المهارات: {data.skills} | التعليم: {data.education}
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. ✏️ تعديل وتخصيص الـ CV لوظيفة معينة
@app.post("/tailor-cv")
def tailor_cv(data: TailorCVRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        أعد صياغة وتحسين السيرة الذاتية التالية لتتطابق تماماً مع وصف الوظيفة المستهدفة باستخدام الكلمات المفتاحية المناسبة:
        وصف الوظيفة: {data.target_job_description}
        السيرة الذاتية: {data.current_cv}
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. 🎤 التحضير للمقابلات الشخصية (STAR Method)
@app.post("/interview-prep")
def interview_prep(data: InterviewPrepRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        بناءً على السيرة الذاتية للوظيفة ({data.job_title}):
        {data.cv_text}
        قم بإنشاء دليل تحضير للمقابلة: أهم 5 أسئلة متوقعة مع إجابات نموذجية بأسلوب STAR وسؤال تقني متقدم.
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 6. ✉️ مولد الخطاب التحفيزي (Cover Letter)
@app.post("/generate-cover-letter")
def generate_cover_letter(data: CoverLetterRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        اكتب خطاب تغطية (Cover Letter) احترافي ومخصص لشركة ({data.company_name}) بناءً على:
        وصف الوظيفة: {data.job_description}
        السيرة الذاتية: {data.cv_text}
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 7. 🎯 استخراج الكلمات المفتاحية لـ ATS
@app.post("/ats-keywords")
def ats_keywords(data: AnalyzeRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        استخرج أهم الكلمات المفتاحية والمهارات الأساسية من وصف الوظيفة التالي:
        {data.job_description}
        ثم قارنها بالسيرة الذاتية وحدد الكلمات الموجودة والكلمات المفقودة التي يجب إضافتها.
        السيرة الذاتية: {data.cv_text}
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 8. 🌐 ترجمة السيرة الذاتية مهنياً
@app.post("/translate-cv")
def translate_cv(data: TranslateRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        ترجم السيرة الذاتية التالية إلى اللغة ({data.target_language}) بشكل مهني واحترافي مع الحفاظ على التنسيق والمصطلحات الوظيفية الصحيحة:
        {data.cv_text}
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 9. 💰 توقع الراتب المالي
@app.post("/predict-salary")
def predict_salary(data: AnalyzeRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        بناءً على الخبرات المذكورة في السيرة الذاتية ووصف الوظيفة التالي:
        وصف الوظيفة: {data.job_description}
        السيرة الذاتية: {data.cv_text}
        قدم تحليلاً لمدى الراتب المتوقع (Minimum & Maximum) والنقاط التي تعزز التفاوض على الراتب.
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 10. 📚 توصيات الكورسات وسد الفجوات
@app.post("/recommend-courses")
def recommend_courses(data: SimpleCVRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        حلل السيرة الذاتية التالية واقترح أهم 3-5 كورسات أو مهارات حديثة ينبغي للمتقدم تعلمها لتطوير مساره المهني وزيادة فرصه:
        {data.cv_text}
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 11. 🌟 تحسين وتلميع النبرة والأسلوب المهني
@app.post("/polish-tone")
def polish_tone(data: SimpleCVRequest):
    try:
        model = get_ai_model()
        prompt = f"""
        قم بمراجعة وصياغة جمل السيرة الذاتية التالية واستبدال الأفعال الضعيفة بأفعال إنجاز قوية (Action Verbs) وتحسين النبرة لتصبح أكثر احترافية وقوة:
        {data.cv_text}
        """
        response = model.generate_content(prompt)
        return {"result": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
