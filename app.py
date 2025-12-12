# ==================== STREAMLIT IMPORTS ====================
import streamlit as st
import json
import re
import time
import fitz  # PyMuPDF
from groq import Groq
import http.client
from datetime import datetime

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="SkillSync AI",
    page_icon="🎯",
    layout="wide"
)

# ==================== CONSTANTS ====================
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 0.1
MAX_TOKENS = 4096
TOP_YOUTUBE_RESULTS = 5
MIN_CV_LENGTH = 100
MIN_JD_LENGTH = 50

# ==================== SIDEBAR - API KEYS ====================
st.sidebar.title("🔑 API Configuration")

# Option 1: Use secrets (for deployment)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
    st.sidebar.success("API keys loaded from secrets")
except:
    # Option 2: Manual input (for local testing)
    st.sidebar.warning("Using manual API input")
    GROQ_API_KEY = st.sidebar.text_input("Groq API Key", type="password")
    SERPER_API_KEY = st.sidebar.text_input("Serper API Key", type="password")

# ==================== MAIN APP ====================
st.title("🎯 SkillSync AI - CV vs Job Analysis")
st.markdown("Upload multiple CVs and compare against a job description")

# ==================== STEP 1: JOB DESCRIPTION INPUT ====================
st.header("📋 Step 1: Enter Job Description")
job_description = st.text_area(
    "Paste the job description here:",
    height=200,
    placeholder="""Paste the complete job description including:
• Requirements
• Responsibilities  
• Qualifications
• Skills needed
• Education requirements""",
    help="Make sure to include all requirements from the job posting"
)

# ==================== STEP 2: UPLOAD MULTIPLE CVs ====================
st.header("📄 Step 2: Upload CVs")
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=['pdf'],
    accept_multiple_files=True,
    help="You can upload multiple CVs at once"
)

# Store CV data in session state
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = []

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        try:
            # Extract text from PDF
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            # Clean text
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,!?\-@()]', ' ', text)
            text = text.strip()
            
            # Validate CV
            cv_errors = []
            if len(text.strip()) < MIN_CV_LENGTH:
                cv_errors.append(f"CV too short ({len(text)} chars)")
            
            cv_lower = text.lower()
            required_sections = ["experience", "education", "skill", "work", "project"]
            found_sections = [section for section in required_sections if section in cv_lower]
            if len(found_sections) < 2:
                cv_errors.append(f"Missing sections")
            
            if cv_errors:
                st.warning(f"⚠️ {uploaded_file.name}: {', '.join(cv_errors)}")
            else:
                # Store valid CV
                st.session_state.cv_data.append({
                    'filename': uploaded_file.name,
                    'text': text[:3000],  # Limit tokens
                    'size': len(text)
                })
                st.success(f"✅ {uploaded_file.name} loaded ({len(text)} chars)")
                
        except Exception as e:
            st.error(f"❌ Error reading {uploaded_file.name}: {str(e)}")

# ==================== STEP 3: ANALYSIS BUTTON ====================
st.header("🤖 Step 3: Analyze")

# Check if ready for analysis
ready_to_analyze = (
    GROQ_API_KEY and 
    SERPER_API_KEY and 
    job_description and 
    len(st.session_state.cv_data) > 0
)

if not ready_to_analyze:
    st.warning("Please complete all steps above before analyzing")
else:
    if st.button("🚀 Analyze All CVs", type="primary", use_container_width=True):
        
        # Store results
        all_results = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Analyze each CV
        for idx, cv in enumerate(st.session_state.cv_data):
            status_text.text(f"Analyzing {cv['filename']}... ({idx+1}/{len(st.session_state.cv_data)})")
            
            # ==================== YOUR EXISTING ANALYSIS CODE ====================
            # (Slightly modified for Streamlit)
            
            # Build prompt (same as your code)
            prompt = f"""
            # EXPERT CV-JOB ANALYSIS REQUEST
            Analyze this CV against the job description with HIGH PRECISION.
            
            ## SPECIFIC REQUIREMENTS:
            1. Return ONLY valid JSON
            2. Include overall_score (0-100)
            3. Include missing_skills array
            4. Include youtube_search_query string
            5. Include cv_improvement_suggestions array
            6. Include missing_education array
            
            CV: {cv['text'][:2500]}
            JD: {job_description[:2500]}
            
            Example JSON format:
            {{
              "overall_score": 85,
              "skills_match": 80,
              "experience_match": 90,
              "education_match": 85,
              "matching_skills": ["Python", "SQL"],
              "missing_skills": ["Kubernetes", "AWS"],
              "missing_education": ["Masters Degree"],
              "cv_improvement_suggestions": [
                {{"section": "Skills", "suggestion": "Add missing skills"}}
              ],
              "youtube_search_query": "Kubernetes tutorial 2024"
            }}
            """
            
            try:
                # Call Groq API (same as your code)
                client = Groq(api_key=GROQ_API_KEY)
                
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a career advisor analyzing CVs."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )
                
                result_text = completion.choices[0].message.content
                result_text = result_text.strip().replace('```json', '').replace('```', '')
                
                result = json.loads(result_text)
                result['analysis_time'] = datetime.now().isoformat()
                
                # YouTube search (same as your code)
                search_query = result.get("youtube_search_query", "")
                if not search_query and result.get("missing_skills"):
                    search_query = f"{result['missing_skills'][0]} tutorial 2024"
                
                if search_query and SERPER_API_KEY:
                    conn = http.client.HTTPSConnection('google.serper.dev')
                    payload = json.dumps({"q": search_query, "num": 3})
                    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
                    conn.request("POST", "/search", payload, headers)
                    response = conn.getresponse()
                    
                    if response.status == 200:
                        data = json.loads(response.read().decode("utf-8"))
                        videos = []
                        for video in data.get("organic", [])[:3]:
                            if "youtube.com" in video.get("link", ""):
                                videos.append({
                                    "title": video.get("title", ""),
                                    "link": video.get("link", ""),
                                    "snippet": video.get("snippet", "")[:150]
                                })
                        result["youtube_videos"] = videos
                
                # Store result
                all_results[cv['filename']] = result
                
            except Exception as e:
                all_results[cv['filename']] = {"error": str(e)}
            
            # Update progress
            progress_bar.progress((idx + 1) / len(st.session_state.cv_data))
        
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        
        # Store in session state
        st.session_state.results = all_results
        
        # Force rerun to show results
        st.rerun()

# ==================== STEP 4: DISPLAY RESULTS ====================
if hasattr(st.session_state, 'results') and st.session_state.results:
    st.header("📊 Analysis Results")
    
    # Create tabs for each CV
    tabs = st.tabs([f"📄 {name}" for name in st.session_state.results.keys()])
    
    for idx, (cv_name, result) in enumerate(st.session_state.results.items()):
        with tabs[idx]:
            
            # Check for errors
            if "error" in result:
                st.error(f"Analysis failed: {result['error']}")
                continue
            
            # Display scores
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Overall Score", f"{result.get('overall_score', 'N/A')}/100")
            with col2:
                st.metric("Skills Match", f"{result.get('skills_match', 'N/A')}/100")
            with col3:
                st.metric("Experience", f"{result.get('experience_match', 'N/A')}/100")
            with col4:
                st.metric("Education", f"{result.get('education_match', 'N/A')}/100")
            
            # Two column layout for skills
            col_left, col_right = st.columns(2)
            
            with col_left:
                # Matching skills
                if result.get("matching_skills"):
                    st.subheader("✅ Matching Skills")
                    for skill in result["matching_skills"][:10]:
                        st.success(f"• {skill}")
                
                # CV improvements
                if result.get("cv_improvement_suggestions"):
                    st.subheader("🔧 CV Improvements")
                    for suggestion in result["cv_improvement_suggestions"][:3]:
                        st.info(f"**{suggestion.get('section', 'General')}**: {suggestion.get('suggestion', '')}")
            
            with col_right:
                # Missing skills
                if result.get("missing_skills"):
                    st.subheader("❌ Missing Skills")
                    for skill in result["missing_skills"][:10]:
                        st.error(f"• {skill}")
                
                # Education gaps
                if result.get("missing_education"):
                    st.subheader("🎓 Education Gaps")
                    for edu in result["missing_education"][:5]:
                        st.warning(f"• {edu}")
            
            # YouTube recommendations
            if result.get("youtube_videos"):
                st.subheader("🎬 Learning Resources")
                for video in result["youtube_videos"]:
                    st.markdown(f"**{video['title']}**")
                    st.markdown(f"[🔗 Watch Video]({video['link']})")
                    if video.get('snippet'):
                        st.caption(video['snippet'])
                    st.divider()
    
    # ==================== COMPARISON TABLE ====================
    st.header("📋 Comparison View")
    
    # Create comparison data
    comparison_rows = []
    for cv_name, result in st.session_state.results.items():
        if "error" not in result:
            comparison_rows.append({
                "CV": cv_name,
                "Score": result.get("overall_score", "N/A"),
                "Missing Skills": len(result.get("missing_skills", [])),
                "Education Gaps": len(result.get("missing_education", [])),
                "Improvements": len(result.get("cv_improvement_suggestions", []))
            })
    
    # Display as table
    if comparison_rows:
        import pandas as pd
        df = pd.DataFrame(comparison_rows)
        st.dataframe(df, use_container_width=True)
        
        # Download button
        json_str = json.dumps(st.session_state.results, indent=2)
        st.download_button(
            label="📥 Download All Results",
            data=json_str,
            file_name="cv_analysis_results.json",
            mime="application/json"
        )

# ==================== FOOTER ====================
st.markdown("---")
st.caption("SkillSync AI | v1.0 | Upload multiple CVs for analysis")