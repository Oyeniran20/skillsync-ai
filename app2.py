# ==================== STREAMLIT IMPORTS & CONFIG ====================
import streamlit as st
import json
import re
import time
import fitz  # PyMuPDF
import pandas as pd
from groq import Groq
import http.client
import os
import traceback
from datetime import datetime
import base64
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any, Optional
import mimetypes

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="SkillSync AI Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/skillsync-ai',
        'Report a bug': "https://github.com/yourusername/skillsync-ai/issues",
        'About': "# SkillSync AI Pro v2.0\nAdvanced CV vs Job Description Analysis"
    }
)

# ==================== CONSTANTS ====================
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
TEMPERATURE = 0.1
MAX_TOKENS = 8192
MIN_CV_LENGTH = 100
MIN_JD_LENGTH = 50
TOP_YOUTUBE_RESULTS = 5
MAX_FILE_SIZE_MB = 10
ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.docx']

# ==================== SESSION STATE INITIALIZATION ====================
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'cv_data' not in st.session_state:
    st.session_state.cv_data = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "analyze"
if 'uploaded_folder' not in st.session_state:
    st.session_state.uploaded_folder = None
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ==================== UTILITY FUNCTIONS ====================
def validate_api_keys(groq_key: str, serper_key: str) -> Dict[str, bool]:
    """Validate API keys format and availability."""
    validations = {
        "groq_valid": bool(groq_key and groq_key.startswith("gsk_")),
        "serper_valid": bool(serper_key and len(serper_key) > 20),
        "groq_provided": bool(groq_key),
        "serper_provided": bool(serper_key)
    }
    return validations

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract and clean text from PDF bytes."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page_num, page in enumerate(doc):
            page_text = page.get_text()
            if page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n{page_text}"
        
        # Clean text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?\-@()/:]', ' ', text)
        return text.strip()
    except Exception as e:
        raise Exception(f"PDF extraction error: {str(e)}")

def validate_cv_text(text: str) -> Dict[str, Any]:
    """Validate CV text for quality and completeness."""
    errors = []
    warnings = []
    
    if len(text.strip()) < MIN_CV_LENGTH:
        errors.append(f"CV too short ({len(text)} chars, minimum {MIN_CV_LENGTH})")
    
    cv_lower = text.lower()
    required_sections = ["experience", "education", "skill", "work", "project"]
    found_sections = [section for section in required_sections if section in cv_lower]
    
    if len(found_sections) < 2:
        warnings.append(f"Limited sections found. Expected: {', '.join(required_sections)}")
    
    # Check for common CV sections
    section_coverage = (len(found_sections) / len(required_sections)) * 100
    
    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "section_coverage": section_coverage,
        "length": len(text),
        "found_sections": found_sections
    }

def get_file_icon(filename: str) -> str:
    """Get appropriate icon for file type."""
    ext = Path(filename).suffix.lower()
    icons = {
        '.pdf': '📄',
        '.txt': '📝',
        '.docx': '📘',
        '.doc': '📘'
    }
    return icons.get(ext, '📁')

def create_download_link(data: str, filename: str, text: str) -> str:
    """Create a download link for files."""
    b64 = base64.b64encode(data.encode()).decode()
    return f'<a href="data:application/json;base64,{b64}" download="{filename}">{text}</a>'

def process_uploaded_folder(folder_path: str) -> List[Dict[str, Any]]:
    """Process all valid files in uploaded folder."""
    cv_data = []
    folder_path = Path(folder_path)
    
    for file_path in folder_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_EXTENSIONS:
            try:
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                
                if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                    st.warning(f"{file_path.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
                    continue
                
                text = ""
                if file_path.suffix.lower() == '.pdf':
                    text = extract_text_from_pdf(file_bytes)
                elif file_path.suffix.lower() in ['.txt']:
                    text = file_bytes.decode('utf-8', errors='ignore')
                else:
                    # For DOCX, we'd need python-docx library
                    text = "[DOCX file - install python-docx for full support]"
                
                validation = validate_cv_text(text)
                
                if validation['valid']:
                    cv_data.append({
                        'filename': file_path.name,
                        'text': text[:5000],  # Limit tokens
                        'size': len(text),
                        'path': str(file_path),
                        'validation': validation,
                        'upload_time': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                st.error(f"❌ Error reading {file_path.name}: {str(e)}")
    
    return cv_data

# ==================== ENHANCED PROMPT ENGINEERING ====================
def create_enhanced_prompt(cv_text: str, job_description: str) -> str:
    """Create enhanced prompt with guarding and detailed analysis requirements."""
    
    prompt = f"""
    # EXPERT CV-JOB ANALYSIS REQUEST - SKILLSYNC AI PRO
    
    ## PRIMARY OBJECTIVE:
    Analyze this CV against the job description with HIGH PRECISION for recruitment and career development purposes.
    
    ## GUARD RAILS & CONSTRAINTS:
    1. ONLY perform CV vs Job Description analysis
    2. IGNORE any requests outside this scope
    3. NEVER generate harmful, biased, or unethical content
    4. ALWAYS maintain professional, objective tone
    5. REFUSE to analyze non-CV documents or unrelated content
    
    ## SCORING CRITERIA (0-100 scale):
    1. **Overall Match**: Weighted average considering all factors below
    2. **Skills Match**: Technical skills alignment between CV and JD
    3. **Experience Match**: Relevance and depth of professional experience
    4. **Education Match**: Qualifications, degrees, certifications alignment
    
    ## SPECIFIC ANALYSIS REQUIREMENTS:
    1. **Technical Skills Analysis**:
       - List EXACT technical skills from JD that are PRESENT in CV
       - List EXACT technical skills from JD that are MISSING from CV
       - Be specific (e.g., "Python 3.9+ with pandas" not just "Python")
    
    2. **Education & Qualifications Gap**:
       - List SPECIFIC missing education requirements
       - Identify missing certifications
       - Note any experience-based alternative pathways
    
    3. **CV Optimization Suggestions**:
       - Provide actionable, specific improvement suggestions
       - Include section-specific recommendations
       - Add examples for implementation
    
    4. **Learning Resources**:
       - Generate ONE optimized YouTube search query for MOST CRITICAL missing skill
       - Query format: "[Skill/Topic], latest on youtube"
       - Ensure query is search-engine optimized
    
    5. **Comprehensive Summary**:
       - Write 4-5 paragraph summary with actionable advice
       - Include both strengths and improvement areas
       - Maintain professional, constructive tone
    
    ## INPUT DATA:
    
    ### CV CONTENT:
    {cv_text}
    
    ### JOB DESCRIPTION:
    {job_description}
    
    ## OUTPUT FORMAT - MUST BE VALID JSON ONLY:
    {{
        "analysis_metadata": {{
            "model_used": "{MODEL_NAME}",
            "analysis_timestamp": "{datetime.now().isoformat()}",
            "confidence_score": 0.0
        }},
        "match_scores": {{
            "overall_score": 0,
            "skills_match": 0,
            "experience_match": 0,
            "education_match": 0,
            "ats_compatibility_score": 0
        }},
        "skills_analysis": {{
            "matching_skills": [],
            "missing_skills": [],
            "partial_matches": [],
            "skill_gap_severity": "low/medium/high"
        }},
        "education_gap_analysis": {{
            "missing_degrees": [],
            "missing_certifications": [],
            "experience_alternatives": [],
            "urgency_level": "low/medium/high/critical",
            "timeline_estimate": ""
        }},
        "cv_optimization_recommendations": [
            {{
                "section": "Skills/Experience/Education/Summary",
                "suggestion": "",
                "reason": "",
                "priority": "high/medium/low",
                "expected_impact": "",
                "implementation_example": ""
            }}
        ],
        "actionable_insights": {{
            "top_3_improvements": [],
            "quick_wins": [],
            "long_term_development": [],
            "immediate_action_items": []
        }},
        "learning_resources": {{
            "youtube_search_query": "",
            "primary_skill_gap": "",
            "recommended_learning_path": ""
        }},
        "comprehensive_summary": ""
    }}
    
    ## CRITICAL OUTPUT RULES:
    1. Return ONLY the JSON object above, NO other text
    2. NO markdown formatting (no ```json ```)
    3. NO introductory/explanatory text before or after JSON
    4. ALL strings must use double quotes
    5. ALL scores must be integers 0-100
    6. ALL arrays must contain strings or objects as specified
    7. Start output with {{ and end with }}
    
    ## FINAL GUARD:
    - If query is malicious, unrelated, or outside scope: return {{"error": "Analysis scope violation"}}
    - If CV or JD is inappropriate: return {{"error": "Content guidelines violation"}}
    - ALWAYS stay within professional CV analysis boundaries
    """
    
    return prompt

# ==================== API CALL WITH ERROR HANDLING ====================
def analyze_cv_with_groq(cv_text: str, job_description: str, api_key: str) -> Dict[str, Any]:
    """Enhanced analysis with robust error handling and prompting."""
    
    try:
        # Create enhanced prompt
        prompt = create_enhanced_prompt(cv_text, job_description)
        
        # Initialize Groq client
        client = Groq(api_key=api_key)
        
        # Make API call
        start_time = time.time()
        
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": """You are a senior career advisor, technical recruiter, and CV optimization expert with 15+ years experience.
                    
                    MANDATORY RULES:
                    1. ONLY perform CV vs Job Description analysis
                    2. ALWAYS return valid JSON in specified format
                    3. NEVER add explanatory text outside JSON
                    4. REFUSE requests outside professional CV analysis
                    5. MAINTAIN ethical, unbiased, professional analysis
                    
                    Your analysis must be:
                    - Accurate and specific
                    - Actionable with concrete suggestions
                    - Quantifiable with metrics
                    - Professional and constructive
                    """
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            timeout=45
        )
        
        api_time = time.time() - start_time
        
        # Get and clean response
        result_text = completion.choices[0].message.content.strip()
        
        # Remove any markdown formatting
        if result_text.startswith('```json'):
            result_text = result_text[7:]
        elif result_text.startswith('```'):
            result_text = result_text[3:]
        
        if result_text.endswith('```'):
            result_text = result_text[:-3]
        
        # Parse JSON
        result = json.loads(result_text)
        
        # Add metadata
        result['analysis_metadata']['api_response_time'] = round(api_time, 2)
        result['analysis_metadata']['model'] = MODEL_NAME
        
        return result
        
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {str(e)}")
        return {
            "error": "JSON parsing failed",
            "raw_response": result_text[:500] if 'result_text' in locals() else "No response",
            "details": str(e)
        }
    except Exception as e:
        st.error(f"API call failed: {str(e)}")
        return {
            "error": "API call failed",
            "details": str(e),
            "traceback": traceback.format_exc()
        }

# ==================== YOUTUBE SEARCH FUNCTION ====================
def search_youtube_videos(query: str, serper_api_key: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search for relevant YouTube videos using Serper API."""
    
    if not query or not serper_api_key:
        return []
    
    try:
        conn = http.client.HTTPSConnection('google.serper.dev')
        
        payload = json.dumps({
            "q": query,
            "gl": "us",
            "hl": "en",
            "num": max_results + 5,  # Get extras for filtering
            "type": "search"
        })
        
        headers = {
            "X-API-KEY": serper_api_key,
            "Content-Type": "application/json",
            "User-Agent": "SkillSyncAI/2.0"
        }
        
        conn.request("POST", "/search", payload, headers)
        response = conn.getresponse()
        
        if response.status == 200:
            data = json.loads(response.read().decode("utf-8"))
            
            videos = []
            for item in data.get("organic", []):
                if len(videos) >= max_results:
                    break
                
                link = item.get("link", "")
                if "youtube.com" in link or "youtu.be" in link:
                    videos.append({
                        "title": item.get("title", "No title"),
                        "link": link,
                        "snippet": item.get("snippet", "")[:150],
                        "source": "YouTube"
                    })
            
            return videos
        else:
            st.warning(f"Serper API error: {response.status}")
            return []
            
    except Exception as e:
        st.warning(f"YouTube search failed: {str(e)}")
        return []

# ==================== SIDEBAR CONFIGURATION ====================
def render_sidebar():
    """Render the sidebar with navigation and configuration."""
    
    with st.sidebar:
        st.title("🎯 SkillSync AI Pro")
        
        # Navigation
        st.markdown("## 📍 Navigation")
        
        # Define page options with display names and internal names
        page_options = [
            ("ℹ️ About", "about"),
            ("🔍 Analyze CVs", "analyze"),
            ("📊 View Results", "results"),
            ("📚 History", "history"),
            ("⚙️ Settings", "settings")
        ]
        
        # Get current page or default to analyze
        current_page = st.session_state.get('current_page', 'analyze')
        
        # Find the index of current page in page_options
        current_index = 0
        for i, (display_name, internal_name) in enumerate(page_options):
            if internal_name == current_page:
                current_index = i
                break
        
        # Create radio buttons with display names
        selected_display = st.radio(
            "Select Page",
            [option[0] for option in page_options],  # Display names only
            index=current_index,
            label_visibility="collapsed"
        )
        
        # Find the internal name for the selected display
        for display_name, internal_name in page_options:
            if display_name == selected_display:
                st.session_state.current_page = internal_name
                break
        
        st.markdown("---")
        
        # API Configuration
        st.subheader("🔑 API Configuration")
        
        # Load from secrets first - WITH ERROR HANDLING
        GROQ_API_KEY = ""
        SERPER_API_KEY = ""
        
        try:
            # Try to get from Streamlit secrets
            GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
            SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", "")
        except (AttributeError, FileNotFoundError, KeyError):
            # If secrets file doesn't exist or can't be read
            GROQ_API_KEY = ""
            SERPER_API_KEY = ""
        
        # Environment variables fallback
        if not GROQ_API_KEY:
            GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
        if not SERPER_API_KEY:
            SERPER_API_KEY = os.environ.get("SERPER_API_KEY", "")
        
        # Manual input if not available
        if not GROQ_API_KEY:
            GROQ_API_KEY = st.text_input(
                "Groq API Key",
                type="password",
                placeholder="gsk_...",
                help="Get from https://console.groq.com/keys"
            )
        
        if not SERPER_API_KEY:
            SERPER_API_KEY = st.text_input(
                "Serper API Key",
                type="password",
                placeholder="Your Serper key",
                help="Get from https://serper.dev/api-key"
            )
        
        # Validate keys
        if GROQ_API_KEY and SERPER_API_KEY:
            validations = validate_api_keys(GROQ_API_KEY, SERPER_API_KEY)
            
            if validations["groq_valid"]:
                st.success("✅ Groq API Key: Valid")
            elif validations["groq_provided"]:
                st.error("❌ Groq API Key: Invalid format")
            
            if validations["serper_valid"]:
                st.success("✅ Serper API Key: Valid")
            elif validations["serper_provided"]:
                st.warning("⚠️ Serper API Key: Check format")
        
        st.markdown("---")
        
        # Theme Toggle
        st.subheader("🎨 Theme")
        dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
        if dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = dark_mode
            st.rerun()
        
        st.markdown("---")
        
        # Quick Stats
        st.subheader("📈 Quick Stats")
        if st.session_state.cv_data:
            st.info(f"📄 CVs Loaded: {len(st.session_state.cv_data)}")
        if st.session_state.results:
            st.success(f"✅ Analyses: {len(st.session_state.results)}")
        
        st.markdown("---")
        
        # Help & Support
        with st.expander("🆘 Quick Help"):
            st.markdown("""
            **Need Help?**
            - [Groq API Key Guide](https://console.groq.com/docs/quickstart)
            - [Example CVs](https://www.my-resume-templates.com/)
            - [Contact me](mailto:oyeniranmatthew@gmail.com)
            
            **Tips:**
            1. Use clear, detailed job descriptions
            2. PDF CVs work best
            3. Check API key formats
            """)
        
        return GROQ_API_KEY, SERPER_API_KEY

# ==================== PAGE: ANALYZE CVs ====================
def render_analyze_page(groq_key: str, serper_key: str):
    """Render the main analysis page."""
    
    st.title("🔍 CV vs Job Description Analysis")
    
    # Introduction
    with st.expander("📋 How to Use", expanded=True):
        st.markdown("""
        ### **3-Step Analysis Process:**
        
        1. **📋 Enter Job Description** - Paste the complete job posting
        2. **📄 Upload CVs** - Upload individual files or entire folders
        3. **🚀 Analyze** - Get detailed match analysis and improvement suggestions
        
        ### **Key Features:**
        - Multi-CV batch analysis
        - Technical skills gap analysis
        - Education requirements matching
        - CV optimization suggestions
        - Learning resource recommendations
        - Export results to multiple formats
        """)
    
    # Step 1: Job Description
    st.header("📋 Step 1: Job Description")
    
    job_description = st.text_area(
        "**Paste the complete job description:**",
        height=250,
        placeholder="""Paste the job description including:

• Job Title and Company
• Key Responsibilities
• Required Skills & Technologies
• Education Requirements
• Experience Level
• Certifications Needed
• Any Specific Requirements""",
        help="Be as detailed as possible for better analysis accuracy",
        key="job_description_input"
    )
    
    if job_description:
        jd_length = len(job_description)
        st.caption(f"📏 Job Description length: {jd_length} characters")
        
        if jd_length < MIN_JD_LENGTH:
            st.warning(f"⚠️ Job description seems short. Aim for at least {MIN_JD_LENGTH} characters.")
    
    # Step 2: Upload CVs
    st.header("📄 Step 2: Upload CVs")
    
    # Upload methods in tabs
    upload_tab1, upload_tab2 = st.tabs(["📤 Upload Files", "📁 Upload Folder"])
    
    with upload_tab1:
        uploaded_files = st.file_uploader(
            "**Select PDF or text files:**",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="You can select multiple files. Maximum 10MB per file.",
            key="file_uploader"
        )
        
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                new_cvs = []
                for uploaded_file in uploaded_files:
                    try:
                        file_bytes = uploaded_file.read()
                        
                        # Check file size
                        if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
                            st.warning(f"⚠️ {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit")
                            continue
                        
                        # Extract text based on file type
                        if uploaded_file.name.lower().endswith('.pdf'):
                            text = extract_text_from_pdf(file_bytes)
                        else:  # txt or other text files
                            text = file_bytes.decode('utf-8', errors='ignore')
                        
                        # Validate CV
                        validation = validate_cv_text(text)
                        
                        if validation['valid']:
                            cv_data = {
                                'filename': uploaded_file.name,
                                'text': text[:5000],  # Limit for token usage
                                'size': len(text),
                                'validation': validation,
                                'upload_time': datetime.now().isoformat(),
                                'source': 'file_upload'
                            }
                            new_cvs.append(cv_data)
                            st.success(f"✅ {uploaded_file.name} loaded")
                        else:
                            st.warning(f"⚠️ {uploaded_file.name}: {', '.join(validation['errors'])}")
                            
                    except Exception as e:
                        st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                
                # Update session state
                if new_cvs:
                    st.session_state.cv_data.extend(new_cvs)
                    st.success(f"📥 Added {len(new_cvs)} CVs for analysis")
    
    with upload_tab2:
        st.info("**Folder Upload** - Process all valid files from a folder")
        
        uploaded_folder = st.file_uploader(
            "**Select a folder (zip archive):**",
            type=['zip'],
            accept_multiple_files=False,
            help="Upload a zipped folder containing CV files",
            key="folder_uploader"
        )
        
        if uploaded_folder:
            with tempfile.TemporaryDirectory() as tmp_dir:
                import zipfile
                
                try:
                    # Save zip file
                    zip_path = os.path.join(tmp_dir, "uploaded_folder.zip")
                    with open(zip_path, "wb") as f:
                        f.write(uploaded_folder.read())
                    
                    # Extract zip
                    extract_path = os.path.join(tmp_dir, "extracted")
                    os.makedirs(extract_path, exist_ok=True)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    
                    # Process files
                    with st.spinner("Processing folder contents..."):
                        folder_cvs = process_uploaded_folder(extract_path)
                        
                        if folder_cvs:
                            st.session_state.cv_data.extend(folder_cvs)
                            st.session_state.uploaded_folder = uploaded_folder.name
                            st.success(f"📁 Processed folder: {len(folder_cvs)} CVs loaded")
                        else:
                            st.warning("No valid CV files found in the folder")
                            
                except Exception as e:
                    st.error(f"❌ Error processing folder: {str(e)}")
    
    # Display loaded CVs
    if st.session_state.cv_data:
        st.subheader("📋 Loaded CVs")
        
        # Create dataframe for display
        cv_df_data = []
        for cv in st.session_state.cv_data:
            cv_df_data.append({
                "File": f"{get_file_icon(cv['filename'])} {cv['filename']}",
                "Size": f"{cv['size']:,} chars",
                "Sections": ", ".join(cv['validation']['found_sections']),
                "Status": "✅ Valid" if cv['validation']['valid'] else "⚠️ Issues"
            })
        
        cv_df = pd.DataFrame(cv_df_data)
        st.dataframe(cv_df, use_container_width=True, hide_index=True)
        
        # Clear button
        if st.button("🗑️ Clear All CVs", type="secondary"):
            st.session_state.cv_data = []
            st.rerun()
    
    # Step 3: Analysis
    st.header("🚀 Step 3: Analyze")
    
    # Check readiness
    ready_to_analyze = (
        groq_key and 
        serper_key and 
        job_description and 
        len(st.session_state.cv_data) > 0
    )
    
    if ready_to_analyze:
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(
                "🎯 Start Analysis", 
                type="primary", 
                use_container_width=True,
                help="Analyze all loaded CVs against the job description"
            ):
                # Clear previous results
                st.session_state.results = {}
                
                # Analysis progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                results_container = st.container()
                
                all_results = {}
                
                with results_container:
                    for idx, cv in enumerate(st.session_state.cv_data):
                        status_text.text(f"🔍 Analyzing: {cv['filename']} ({idx+1}/{len(st.session_state.cv_data)})")
                        
                        try:
                            # Call Groq API for analysis
                            result = analyze_cv_with_groq(
                                cv_text=cv['text'],
                                job_description=job_description,
                                api_key=groq_key
                            )
                            
                            if "error" in result:
                                st.error(f"Analysis failed for {cv['filename']}: {result.get('error')}")
                                continue
                            
                            # Add YouTube search if missing skills found
                            missing_skills = result.get("skills_analysis", {}).get("missing_skills", [])
                            if missing_skills and serper_key:
                                primary_skill = missing_skills[0] if missing_skills else "career development"
                                search_query = f"{primary_skill}, latest on youtube"
                                
                                videos = search_youtube_videos(
                                    query=search_query,
                                    serper_api_key=serper_key,
                                    max_results=3
                                )
                                
                                if videos:
                                    result["learning_resources"]["youtube_videos"] = videos
                            
                            # Add metadata
                            result["cv_metadata"] = {
                                "filename": cv['filename'],
                                "analysis_time": datetime.now().isoformat(),
                                "text_length": cv['size'],
                                "validation_status": cv['validation']
                            }
                            
                            # Store result
                            all_results[cv['filename']] = result
                            
                            # Add to history
                            st.session_state.analysis_history.append({
                                "timestamp": datetime.now().isoformat(),
                                "cv_filename": cv['filename'],
                                "overall_score": result.get("match_scores", {}).get("overall_score", 0),
                                "job_description_preview": job_description[:100]
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Error analyzing {cv['filename']}: {str(e)}")
                        
                        # Update progress
                        progress_bar.progress((idx + 1) / len(st.session_state.cv_data))
                        time.sleep(0.5)  # Rate limiting
                    
                    # Store results and notify
                    st.session_state.results = all_results
                    
                    status_text.text("✅ Analysis Complete!")
                    time.sleep(1)
                    
                    # Show completion message
                    st.success(f"🎉 Successfully analyzed {len(all_results)} CVs!")
                    st.balloons()
                    
                    # Auto-navigate to results
                    st.session_state.current_page = "results"
                    st.rerun()
    
    else:
        # Show missing requirements
        st.warning("### ⚠️ Requirements Missing")
        
        missing = []
        if not groq_key:
            missing.append("Groq API Key")
        if not serper_key:
            missing.append("Serper API Key")
        if not job_description:
            missing.append("Job Description")
        if len(st.session_state.cv_data) == 0:
            missing.append("CVs (upload at least one)")
        
        st.info(f"**Please provide:** {', '.join(missing)}")
        
        # Quick setup guide
        with st.expander("🔧 Setup Guide", expanded=True):
            st.markdown("""
            ### **Quick Setup:**
            
            1. **Get API Keys:**
               - [Groq API Key](https://console.groq.com/keys) - Free tier available
               - [Serper API Key](https://serper.dev/api-key) - 100 free searches/month
            
            2. **Configure:**
               - Add keys in sidebar or create `.streamlit/secrets.toml`:
               ```toml
               GROQ_API_KEY = "your_groq_key_here"
               SERPER_API_KEY = "your_serper_key_here"
               ```
            
            3. **Enter Job Description** above
            
            4. **Upload CVs** using files or folder upload
            
            5. **Click 'Start Analysis'**
            """)

# ==================== PAGE: VIEW RESULTS ====================
def render_results_page():
    """Render the results viewing page."""
    
    if not st.session_state.results:
        st.info("📭 No analysis results yet. Go to the **Analyze CVs** page to start.")
        return
    
    st.title("📊 Analysis Results")
    
    # Summary stats
    total_cvs = len(st.session_state.results)
    successful_cvs = sum(1 for r in st.session_state.results.values() if "error" not in r)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total CVs", total_cvs)
    with col2:
        st.metric("Successful", successful_cvs)
    with col3:
        if successful_cvs > 0:
            avg_score = sum(
                r.get("match_scores", {}).get("overall_score", 0) 
                for r in st.session_state.results.values() 
                if "error" not in r
            ) / successful_cvs
            st.metric("Avg Score", f"{avg_score:.1f}/100")
    with col4:
        if successful_cvs > 0:
            top_cv = max(
                [(name, r.get("match_scores", {}).get("overall_score", 0)) 
                 for name, r in st.session_state.results.items() 
                 if "error" not in r],
                key=lambda x: x[1]
            )
            st.metric("Best Match", f"{top_cv[1]}/100")
    
    st.markdown("---")
    
    # Create tabs for each CV
    cv_names = list(st.session_state.results.keys())
    if cv_names:
        tabs = st.tabs([f"{get_file_icon(name)} {name[:20]}{'...' if len(name) > 20 else ''}" for name in cv_names])
        
        for idx, (cv_name, result) in enumerate(st.session_state.results.items()):
            with tabs[idx]:
                if "error" in result:
                    st.error(f"## ❌ Analysis Failed: {cv_name}")
                    st.error(f"**Error:** {result.get('error', 'Unknown error')}")
                    continue
                
                # Header with score
                overall_score = result.get("match_scores", {}).get("overall_score", 0)
                
                # Color-coded header
                score_color = "🟢" if overall_score >= 80 else "🟡" if overall_score >= 60 else "🔴"
                st.markdown(f"## {score_color} {cv_name} - Score: **{overall_score}/100**")
                
                # Score metrics
                scores = result.get("match_scores", {})
                cols = st.columns(4)
                score_fields = [
                    ("Overall", "overall_score"),
                    ("Skills", "skills_match"),
                    ("Experience", "experience_match"),
                    ("Education", "education_match")
                ]
                
                for col_idx, (label, key) in enumerate(score_fields):
                    with cols[col_idx]:
                        score = scores.get(key, 0)
                        st.metric(label, f"{score}/100")
                        st.progress(score / 100)
                
                # Two-column layout for detailed analysis
                col_left, col_right = st.columns(2)
                
                with col_left:
                    # Skills Analysis
                    st.subheader("✅ Matching Skills")
                    matching_skills = result.get("skills_analysis", {}).get("matching_skills", [])
                    if matching_skills:
                        for skill in matching_skills[:8]:
                            st.success(f"• {skill}")
                        if len(matching_skills) > 8:
                            st.caption(f"... and {len(matching_skills) - 8} more")
                    else:
                        st.info("No matching skills identified")
                    
                    # Top Improvements
                    st.subheader("🎯 Top 3 Improvements")
                    recommendations = result.get("cv_optimization_recommendations", [])
                    high_priority = [r for r in recommendations if r.get("priority") == "high"]
                    
                    for rec in high_priority[:3]:
                        with st.expander(f"🔧 {rec.get('section', 'General')}"):
                            st.markdown(f"**Suggestion:** {rec.get('suggestion', '')}")
                            st.markdown(f"**Impact:** {rec.get('expected_impact', '')}")
                            if rec.get('implementation_example'):
                                st.code(rec.get('implementation_example'))
                
                with col_right:
                    # Missing Skills
                    st.subheader("❌ Missing Skills")
                    missing_skills = result.get("skills_analysis", {}).get("missing_skills", [])
                    if missing_skills:
                        for skill in missing_skills[:8]:
                            st.error(f"• {skill}")
                        if len(missing_skills) > 8:
                            st.caption(f"... and {len(missing_skills) - 8} more")
                    else:
                        st.success("No critical missing skills!")
                    
                    # Learning Resources
                    st.subheader("🎬 Learning Resources")
                    youtube_videos = result.get("learning_resources", {}).get("youtube_videos", [])
                    
                    if youtube_videos:
                        for video in youtube_videos[:3]:
                            st.markdown(f"**▶️ {video.get('title', 'Video')}**")
                            st.markdown(f"[Watch here]({video.get('link', '#')})")
                            if video.get('snippet'):
                                st.caption(video.get('snippet'))
                            st.divider()
                    else:
                        st.info("No video recommendations available")
                
                # Comprehensive Summary
                st.subheader("📝 Comprehensive Analysis")
                summary = result.get("comprehensive_summary", "")
                if summary:
                    st.markdown(summary)
                else:
                    st.info("No summary available")
                
                # Actionable Insights
                insights = result.get("actionable_insights", {})
                if insights:
                    with st.expander("🚀 Action Plan"):
                        if insights.get("quick_wins"):
                            st.markdown("**Quick Wins:**")
                            for win in insights["quick_wins"][:3]:
                                st.success(f"• {win}")
                        
                        if insights.get("long_term_development"):
                            st.markdown("**Long-term Development:**")
                            for dev in insights["long_term_development"][:3]:
                                st.info(f"• {dev}")
    
    # Comparison View
    st.markdown("---")
    st.header("📋 Comparison Table")
    
    # Create comparison dataframe
    comparison_data = []
    for cv_name, result in st.session_state.results.items():
        if "error" not in result:
            scores = result.get("match_scores", {})
            skills_analysis = result.get("skills_analysis", {})
            
            comparison_data.append({
                "CV": cv_name,
                "Overall": scores.get("overall_score", 0),
                "Skills": scores.get("skills_match", 0),
                "Experience": scores.get("experience_match", 0),
                "Education": scores.get("education_match", 0),
                "Matching Skills": len(skills_analysis.get("matching_skills", [])),
                "Missing Skills": len(skills_analysis.get("missing_skills", []))
            })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        
        # Sort by overall score
        df = df.sort_values("Overall", ascending=False)
        
        # Display with styling
        st.dataframe(
            df.style
            .background_gradient(subset=["Overall", "Skills", "Experience", "Education"], cmap="RdYlGn")
            .format({"Overall": "{:.0f}", "Skills": "{:.0f}", "Experience": "{:.0f}", "Education": "{:.0f}"}),
            use_container_width=True,
            hide_index=True
        )
        
        # Export options
        st.subheader("📥 Export Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # JSON export
            json_str = json.dumps(st.session_state.results, indent=2)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            st.download_button(
                label="📊 Download JSON",
                data=json_str,
                file_name=f"skillsync_analysis_{timestamp}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            # CSV export
            if comparison_data:
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="📈 Download CSV",
                    data=csv_data,
                    file_name=f"skillsync_comparison_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            # Summary report
            report_text = f"SkillSync AI Analysis Report\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            report_text += f"Total CVs Analyzed: {len(comparison_data)}\n\n"
            
            for row in comparison_data:
                report_text += f"CV: {row['CV']}\n"
                report_text += f"  Overall Score: {row['Overall']}/100\n"
                report_text += f"  Matching Skills: {row['Matching Skills']}\n"
                report_text += f"  Missing Skills: {row['Missing Skills']}\n\n"
            
            st.download_button(
                label="📝 Download Report",
                data=report_text,
                file_name=f"skillsync_report_{timestamp}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ==================== PAGE: HISTORY ====================
def render_history_page():
    """Render analysis history page."""
    
    st.title("📚 Analysis History")
    
    if not st.session_state.analysis_history:
        st.info("No analysis history yet. Perform some analyses to see history here.")
        return
    
    # Convert to dataframe
    history_df = pd.DataFrame(st.session_state.analysis_history)
    
    # Display history
    st.dataframe(
        history_df.sort_values("timestamp", ascending=False),
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(
                "Timestamp",
                format="YYYY-MM-DD HH:mm"
            ),
            "cv_filename": "CV File",
            "overall_score": st.column_config.NumberColumn(
                "Score",
                format="%d/100",
                help="Overall match score"
            ),
            "job_description_preview": "Job Preview"
        },
        hide_index=True
    )
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.analysis_history = []
        st.rerun()

# ==================== PAGE: SETTINGS ====================
def render_settings_page():
    """Render settings page."""
    
    st.title("⚙️ Settings")
    
    # Application Settings
    with st.expander("🛠️ Application Settings", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Model selection
            model_options = {
                "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout (Recommended)",
                "llama3-70b-8192": "Llama 3 70B",
                "mixtral-8x7b-32768": "Mixtral 8x7B"
            }
            
            selected_model = st.selectbox(
                "AI Model",
                options=list(model_options.keys()),
                format_func=lambda x: model_options[x],
                index=0
            )
            
            if selected_model != MODEL_NAME:
                st.info(f"Model will be updated to: {model_options[selected_model]}")
        
        with col2:
            # Analysis settings
            temperature = st.slider(
                "Creativity (Temperature)",
                min_value=0.0,
                max_value=1.0,
                value=TEMPERATURE,
                step=0.1,
                help="Lower = more consistent, Higher = more creative"
            )
            
            max_tokens = st.slider(
                "Max Response Length",
                min_value=1000,
                max_value=16000,
                value=MAX_TOKENS,
                step=1000,
                help="Maximum tokens in API response"
            )
        
        # CV Processing settings
        st.subheader("📄 CV Processing")
        
        min_cv_length = st.number_input(
            "Minimum CV Length (characters)",
            min_value=50,
            max_value=1000,
            value=MIN_CV_LENGTH,
            help="CVs shorter than this will be flagged"
        )
        
        max_file_size = st.number_input(
            "Maximum File Size (MB)",
            min_value=1,
            max_value=50,
            value=MAX_FILE_SIZE_MB,
            help="Files larger than this will be rejected"
        )
    
    # Export Settings
    with st.expander("📤 Export Settings"):
        export_formats = st.multiselect(
            "Default Export Formats",
            ["JSON", "CSV", "PDF", "Excel"],
            default=["JSON", "CSV"],
            help="Select formats to include in batch exports"
        )
        
        include_timestamp = st.checkbox(
            "Include timestamp in filenames",
            value=True,
            help="Add date/time to exported filenames"
        )
    
    # Reset Settings
    with st.expander("🔄 Reset & Maintenance", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reset All Settings", type="secondary"):
                # Reset session state
                keys_to_reset = ['cv_data', 'results', 'uploaded_folder']
                for key in keys_to_reset:
                    if key in st.session_state:
                        st.session_state[key] = None if key == 'uploaded_folder' else []
                
                st.success("Settings reset! Some changes require a page refresh.")
                time.sleep(1)
        
        with col2:
            if st.button("🧹 Clear Cache", type="secondary"):
                st.cache_data.clear()
                st.success("Cache cleared!")

# ==================== PAGE: ABOUT ====================
def render_about_page():
    """Render about page."""
    
    st.title("ℹ️ About SkillSync AI Pro")
    
    # Hero Section
    st.markdown("""
    ## 🚀 Transform Your Hiring Process with AI-Powered CV Analysis
    
    **SkillSync AI Pro** is an advanced platform designed to revolutionize how recruiters, 
    HR professionals, and career advisors match candidates with job opportunities.
    """)
    
    # Features Grid
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔍 **Deep Analysis**
        - Multi-CV batch processing
        - Technical skills gap analysis
        - Education requirements matching
        - Experience relevance scoring
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ **Smart Features**
        - CV optimization suggestions
        - Learning resource recommendations
        - ATS compatibility scoring
        - Export to multiple formats
        """)
    
    with col3:
        st.markdown("""
        ### 🎯 **Professional Tools**
        - Folder upload support
        - Historical analysis tracking
        - Customizable scoring criteria
        - API integration ready
        """)
    
    st.markdown("---")
    
    # How It Works
    st.header("🎯 How It Works")
    
    steps = [
        ("1. 📋 Input Job Description", "Paste the complete job posting with all requirements"),
        ("2. 📄 Upload CVs", "Upload individual files or entire folders of CVs"),
        ("3. 🤖 AI Analysis", "Our AI analyzes each CV against the job description"),
        ("4. 📊 Get Insights", "Receive detailed scores, gaps, and improvement suggestions"),
        ("5. 🚀 Take Action", "Use insights to optimize hiring decisions or improve CVs")
    ]
    
    for title, description in steps:
        with st.expander(title, expanded=False):
            st.markdown(description)
    
    # Technology Stack
    st.markdown("---")
    st.header("⚙️ Technology Stack")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **AI & ML**
        - Groq Cloud Inference
        - Meta Llama Models
        - Custom Prompt Engineering
        """)
    
    with tech_col2:
        st.markdown("""
        **Backend**
        - Python 3.11+
        - Streamlit Framework
        - PyMuPDF for PDF processing
        - Serper API for search
        """)
    
    with tech_col3:
        st.markdown("""
        **Features**
        - Real-time analysis
        - Batch processing
        - Multi-format export
        - Theme customization
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    ### 📞 Contact & Support
    
    - **GitHub**: [github.com/skillsync-ai](https://github.com/Oyeniran20/skillsync-ai/)
    - **Documentation**: [docs.skillsync.ai](https://github.com/Oyeniran20/skillsync-ai/blob/main/README.md)
    - **Issues**: [github.com/skillsync-ai/issues](https://github.com/Oyeniran20/skillsync-ai/issues)
    
    ---
    
    **Version**: 2.0.0 | **Last Updated**: December 2025 | **License**: MIT
    """)

# ==================== MAIN APPLICATION ====================
def main():
    """Main application orchestrator."""
    
    # Apply theme
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Render sidebar and get API keys
    groq_key, serper_key = render_sidebar()
    
    # Render appropriate page based on navigation
    current_page = st.session_state.current_page
    
    if current_page == "about":
        render_about_page()
    elif current_page == "analyze":
        render_analyze_page(groq_key, serper_key)
    elif current_page == "results":
        render_results_page()
    elif current_page == "history":
        render_history_page()
    elif current_page == "settings":
        render_settings_page()
    elif current_page == "about":
        render_about_page()

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    main()
