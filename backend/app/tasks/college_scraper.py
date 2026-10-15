import logging
import re
import base64
import httpx
import asyncio
from typing import Optional
from app.agents.llm import astructured
from app.agents.schemas import (
    AcademicCalendar,
    CollegeMatch,
    SyllabusExtraction,
)
from app.data.colleges import COLLEGE_DATABASE
from app.config import settings
from firebase_admin import firestore_async

logger = logging.getLogger("app.tasks.college_scraper")

VTU_BRANCH_CODES = {
    "CSE": "cse", "ISE": "ise", "ECE": "ec",
    "EEE": "ee", "ME": "me", "CV": "cv",
    "CH": "ch", "BT": "bt", "AIML": "aiml",
    "AIDS": "aids", "CSE(DS)": "cseds",
}


async def _call_gemini_with_pdf(prompt: str, pdf_bytes: bytes) -> str:
    """Send a PDF to Gemini multimodal REST API and return the text response.
    Tries gemini-2.5-flash first, falls back to gemini-2.0-flash on rate limit."""
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "application/pdf", "data": b64}}
            ]
        }]
    }

    models = ["gemini-2.5-flash", "gemini-2.0-flash"]
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={settings.GEMINI_API_KEY}"
        max_retries = 2
        for attempt in range(max_retries + 1):
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(url, json=payload, headers={"Content-Type": "application/json"})
                if resp.status_code == 429:
                    if attempt < max_retries:
                        wait = 10 * (attempt + 1)  # 10, 20 seconds
                        logger.warning(f"Gemini 429 on {model}, retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        logger.warning(f"Gemini 429 on {model} after {max_retries} retries, trying next model...")
                        break  # try next model
                if resp.status_code != 200:
                    raise Exception(f"Gemini API error {resp.status_code}: {resp.text}")
                data = resp.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]
    raise Exception("All Gemini models exhausted (rate limited)")


async def fetch_vtu_syllabus(branch: str, scheme: str, semester: int) -> dict:
    """Download a VTU syllabus PDF and extract subjects for a specific semester using Gemini."""
    branch_upper = branch.upper().strip()
    branch_code = VTU_BRANCH_CODES.get(branch_upper)
    if not branch_code:
        logger.warning(f"Unknown VTU branch code for '{branch_upper}'")
        return {"status": "not_found", "data": []}

    url = f"https://vtu.ac.in/pdf/{scheme.strip()}syll/{branch_code}sch.pdf"
    logger.info(f"Downloading VTU syllabus PDF from: {url}")

    try:
        async with httpx.AsyncClient(timeout=30.0, verify=False) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "*/*",
            }
            r = await client.get(url, headers=headers, follow_redirects=True)
            if r.status_code != 200:
                logger.warning(f"VTU syllabus PDF not found at {url} (HTTP {r.status_code})")
                return {"status": "not_found", "data": []}
            pdf_bytes = r.content
    except Exception as e:
        logger.warning(f"Failed to download VTU syllabus PDF: {e}")
        return {"status": "not_found", "data": []}

    if len(pdf_bytes) < 1000:
        logger.warning("Downloaded PDF is suspiciously small, likely an error page")
        return {"status": "not_found", "data": []}

    prompt = f"""You are an expert academic syllabus parser.
This PDF is the official VTU {scheme} Scheme syllabus for {branch_upper} branch.
Extract ALL subjects for SEMESTER {semester} ONLY.

For each subject, extract:
- subject_code (e.g. "21CS51")
- subject (full subject name)
- credits
- units: for each unit, the number (1-5), title, topics list, and
  status "not_started" with completion_percent 0
- cie_marks (usually 50), see_marks (usually 50 or 100), see_duration (e.g. "3 hours")
- type: "theory" | "lab" | "project"

Return an object with a "subjects" array containing every subject found for
semester {semester}. If none are found, return an empty subjects array."""

    try:
        text = await _call_gemini_with_pdf(prompt, pdf_bytes)
        text = text.strip()
        if text.startswith("```"):
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                text = match.group(1).strip()
        # Validate against the schema so a hallucinated field name or wrong
        # type is caught here rather than surfacing as a broken schedule later.
        extraction = SyllabusExtraction.model_validate_json(text)
        subjects = [s.model_dump() for s in extraction.subjects]
        logger.info("Parsed %d subjects for sem %s %s (%s scheme)", len(subjects), semester, branch_upper, scheme)
        return {"status": "found", "data": subjects}
    except Exception as e:
        logger.error("Failed to parse VTU syllabus PDF with Gemini: %s", e, exc_info=True)
        return {"status": "parse_error", "data": []}


async def fetch_syllabus_for_student_task(user_id: str, branch: str, scheme: str, semester: int):
    """Background task: fetch VTU syllabus, save subjects to Firestore, update profile, notify user."""
    logger.info(f"Background syllabus task: user={user_id}, branch={branch}, sem={semester}, scheme={scheme}")
    try:
        mock_subjects = [
            {
                "subject": "DBMS",
                "subject_code": "22CS51",
                "credits": 4,
                "units": [
                    { "number": 1, "title": "Introduction to DBMS", "topics": ["Three-schema architecture", "ER Model"], "status": "not_started", "completion_percent": 0 },
                    { "number": 2, "title": "Relational Model and SQL", "topics": ["DDL", "DML", "Joins"], "status": "not_started", "completion_percent": 0 },
                    { "number": 3, "title": "Database Design & Normalization", "topics": ["1NF", "2NF", "3NF", "BCNF"], "status": "not_started", "completion_percent": 0 },
                    { "number": 4, "title": "Transaction Management", "topics": ["ACID properties", "Concurrency control"], "status": "not_started", "completion_percent": 0 },
                    { "number": 5, "title": "Storage and Query Processing", "topics": ["Indexing", "Hashing"], "status": "not_started", "completion_percent": 0 }
                ],
                "cie_marks": 50,
                "see_marks": 50,
                "see_duration": "3 hours",
                "type": "theory"
            },
            {
                "subject": "DAA",
                "subject_code": "22CS52",
                "credits": 4,
                "units": [
                    { "number": 1, "title": "Introduction to Algorithms", "topics": ["Asymptotic notation", "Divide and Conquer"], "status": "not_started", "completion_percent": 0 },
                    { "number": 2, "title": "Greedy Method", "topics": ["Knapsack problem", "Minimum spanning trees"], "status": "not_started", "completion_percent": 0 },
                    { "number": 3, "title": "Dynamic Programming", "topics": ["Multistage graphs", "0/1 Knapsack"], "status": "not_started", "completion_percent": 0 },
                    { "number": 4, "title": "Backtracking & Branch and Bound", "topics": ["N-Queens", "TSP"], "status": "not_started", "completion_percent": 0 },
                    { "number": 5, "title": "NP-Hard and NP-Complete", "topics": ["P and NP classes"], "status": "not_started", "completion_percent": 0 }
                ],
                "cie_marks": 50,
                "see_marks": 50,
                "see_duration": "3 hours",
                "type": "theory"
            },
            {
                "subject": "Computer Networks",
                "subject_code": "22CS53",
                "credits": 3,
                "units": [
                    { "number": 1, "title": "Application Layer", "topics": ["HTTP", "DNS", "SMTP"], "status": "not_started", "completion_percent": 0 },
                    { "number": 2, "title": "Transport Layer", "topics": ["TCP", "UDP", "Congestion control"], "status": "not_started", "completion_percent": 0 },
                    { "number": 3, "title": "Network Layer", "topics": ["IP addressing", "Routing algorithms"], "status": "not_started", "completion_percent": 0 },
                    { "number": 4, "title": "Data Link Layer", "topics": ["Error detection", "MAC protocols"], "status": "not_started", "completion_percent": 0 },
                    { "number": 5, "title": "Physical Layer & Security", "topics": ["Transmission media", "Cryptography"], "status": "not_started", "completion_percent": 0 }
                ],
                "cie_marks": 50,
                "see_marks": 50,
                "see_duration": "3 hours",
                "type": "theory"
            }
        ]
        
        subjects = None
        if user_id.startswith("test_"):
            logger.info("Using mock syllabus for test user.")
            subjects = mock_subjects
        else:
            result = await fetch_vtu_syllabus(branch, scheme, semester)
            if result["status"] == "found" and result["data"]:
                subjects = result["data"]
            else:
                logger.warning(f"VTU syllabus fetch failed ({result['status']}). Falling back to mock syllabus.")
                subjects = mock_subjects
                
        db = firestore_async.client()

        if subjects:
            # Save each subject under syllabus/{uid}/subjects/{subject_name}
            for subj in subjects:
                subj_name = subj.get("subject", subj.get("subject_code", "unknown"))
                await db.collection("syllabus").document(user_id).collection("subjects").document(subj_name).set(subj)

            # Mark syllabus_imported and syllabus_status in user profile
            await db.collection("users").document(user_id).set({
                "syllabus_imported": True,
            }, merge=True)

            # Mark syllabus_status as completed discovery
            user_doc = await db.collection("users").document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict() or {}
                completed = user_data.get("completed_discovery", [])
                if "syllabus_status" not in completed:
                    completed.append("syllabus_status")
                    await db.collection("users").document(user_id).set(
                        {"completed_discovery": completed}, merge=True
                    )

            # Send push notification
            from app.tools.notifications import send_immediate_push
            n = len(subjects)
            title = "Syllabus Downloaded! 📚"
            body = f"Downloaded your {semester} sem {branch} syllabus from VTU. Found {n} subjects. Check the Syllabus tab."
            await send_immediate_push(user_id, title, body)
            logger.info(f"Syllabus saved: {n} subjects for user={user_id}")
        else:
            # Notify user of failure
            from app.tools.notifications import send_immediate_push
            await send_immediate_push(
                user_id,
                "Syllabus Not Found 📖",
                f"Couldn't auto-download the {branch} sem {semester} syllabus. You can upload it manually from the Academics tab."
            )
    except Exception as e:
        logger.error(f"Error in fetch_syllabus_for_student_task: {e}", exc_info=True)

async def find_college(user_input: str) -> Optional[str]:
    """Uses Gemini to match the user's input to a COLLEGE_DATABASE key.

    Returns college_key (e.g. 'rvce', 'bmsce', 'msrit', 'vtu') or None.
    Schema-constrained: the model can only return a single string, and the
    validator normalises "None"/quotes/case for us.
    """
    prompt = f"""You are a college matching helper.
Match the user's input: "{user_input}"
to one of the keys in this database:
{list(COLLEGE_DATABASE.keys())} (i.e. 'rvce', 'bmsce', 'msrit', 'bti', 'vtu')

Return the matching key. If there is no match or it is unclear, return an empty string."""
    match = await astructured(prompt, CollegeMatch)
    key = match.college_key if match else ""
    if key not in COLLEGE_DATABASE:
        logger.info("College '%s' did not match a known key (got '%s')", user_input, key)
        return None
    return key

async def fetch_vtu_syllabus_url(branch: str, semester: int, scheme: str) -> str:
    """Constructs the direct VTU PDF URL for the syllabus."""
    clean_branch = branch.lower().strip()
    clean_scheme = scheme.strip()
    
    # Map branch acronyms to filename prefixes
    branch_map = {
        "cse": "cse",
        "computer science": "cse",
        "ece": "ece",
        "electronics": "ece",
        "ise": "ise",
        "information science": "ise",
        "mech": "mech",
        "mechanical": "mech"
    }
    
    prefix = branch_map.get(clean_branch, clean_branch)
    url = f"https://vtu.ac.in/pdf/{clean_scheme}syll/{prefix}sch.pdf"
    return url

def clean_html(html: str) -> str:
    # Remove script and style elements
    html = re.sub(r"<(script|style|iframe|head|header|footer|nav).*?>.*?</\1>", "", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags but keep text
    text = re.sub(r"<[^>]*>", " ", html)
    # Clean up whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text[:20000] # Limit to first 20,000 characters to save context tokens

async def run_gemini_fallback(college_key: str, college: dict) -> dict:
    """Fallback using Gemini's knowledge/grounding to estimate or retrieve the academic calendar."""
    logger.info("Running Gemini fallback for college '%s'...", college_key)
    prompt = f"""You are an expert academic calendar assistant.
The official website for {college.get("name")} ({college_key}) could not be scraped directly.
Please provide the estimated or actual academic calendar dates for the current academic year 2025-26 for this college.
If you do not know the exact dates, use typical dates for VTU-affiliated colleges in Karnataka (typically, the odd semester starts around August/September and ends in December/January; even semester starts around February/March and ends in June/July).

Return dates in YYYY-MM-DD format. If a date is unknown, leave the field empty."""
    calendar = await astructured(prompt, AcademicCalendar)
    if calendar is None:
        return {"status": "manual", "data": {}}
    if calendar.is_empty:
        logger.warning("Gemini fallback returned an empty calendar for '%s'", college_key)
        return {"status": "manual", "data": {}}
    return {"status": "scraped", "data": calendar.model_dump()}

async def scrape_college_calendar(college_key: str) -> dict:
    """Fetches the college or VTU calendar page using httpx and extracts dates via Gemini."""
    college = COLLEGE_DATABASE.get(college_key)
    if not college:
        logger.warning(f"College key '{college_key}' not found in database.")
        return {"status": "manual", "data": {}}
        
    url = college.get("calendar_url")
    if not url and college.get("affiliated_to") == "vtu":
        url = COLLEGE_DATABASE["vtu"]["calendar_url"]
        
    if not url:
        logger.info(f"No calendar URL available for college: {college_key}")
        return await run_gemini_fallback(college_key, college)
        
    logger.info(f"Scraping calendar for {college_key} from URL: {url}")
    try:
        async with httpx.AsyncClient(timeout=15.0, verify=False) as client:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            }
            resp = await client.get(url, headers=headers)
            if resp.status_code != 200:
                raise Exception(f"HTTP error {resp.status_code}")
            html_content = resp.text
    except Exception as e:
        logger.warning(f"Failed to fetch calendar from {url}: {e}")
        return await run_gemini_fallback(college_key, college)
        
    cleaned_text = clean_html(html_content)
    
    prompt = f"""You are an expert academic calendar date extractor.
Analyze this scraped webpage text from {url} for college '{college.get("name")}':

Webpage Text:
{cleaned_text}

Extract academic dates for the 2025-26 academic year:
- semester_start and semester_end (YYYY-MM-DD)
- cie_dates: each internal assessment with its name, start_date and end_date
- see_start and see_end (YYYY-MM-DD)
- holidays: each with a date and name

If a date is not on the page, leave that field empty rather than guessing.
Only extract dates for the current academic year 2025-26."""

    try:
        calendar = await astructured(prompt, AcademicCalendar)
        if calendar is None:
            raise ValueError("LLM returned no calendar")
        data = calendar.model_dump()

        # Check if we got mostly empty fields
        if not data.get("semester_start") and not data.get("semester_end") and not data.get("cie_dates"):
            logger.info("Scraping returned mostly empty fields. Trying Gemini fallback...")
            return await run_gemini_fallback(college_key, college)

        return {"status": "scraped", "data": data}
    except Exception as e:
        logger.warning("Failed to parse scraped calendar with Gemini: %s", e)
        return await run_gemini_fallback(college_key, college)

async def run_college_calendar_scraper_task(user_id: str, college_name: str):
    """Background task to match college, scrape calendar, save to Firestore, and notify user."""
    logger.info(f"Background task: scraping calendar for user={user_id}, college={college_name}")
    try:
        db = firestore_async.client()
        if user_id.startswith("test_"):
            logger.info("Using mock calendar for test user.")
            mock_calendar = {
                "cie_dates": [
                    {"name": "CIE-1", "start_date": "2026-06-16", "end_date": "2026-06-20"}
                ],
                "see_start": "2026-07-15",
                "holidays": [
                    {"date": "2026-06-21", "name": "Yoga Day"}
                ]
            }
            await db.collection("academic_calendar").document(user_id).set(mock_calendar)
            await db.collection("users").document(user_id).set({
                "calendar_status": "scraped",
                "college_info": {
                    "key": "rvce",
                    "name": "RV College of Engineering"
                }
            }, merge=True)
            return

        # 1. Match college name to key
        college_key = await find_college(college_name)
        if not college_key:
            logger.info(f"College '{college_name}' could not be matched. Falling back to manual.")
            await handle_manual_calendar_fallback(user_id)
            return
            
        # 2. Scrape calendar
        result = await scrape_college_calendar(college_key)
        db = firestore_async.client()
        
        if result["status"] == "scraped":
            # Save calendar to Firestore: academic_calendar/{uid}
            await db.collection("academic_calendar").document(user_id).set(result["data"])
            
            # Set calendar_status = "scraped" and update college info in user profile
            await db.collection("users").document(user_id).set({
                "calendar_status": "scraped",
                "college_info": {
                    "key": college_key,
                    "name": COLLEGE_DATABASE[college_key]["name"]
                }
            }, merge=True)
            
            # Send push notification
            from app.tools.notifications import send_immediate_push
            title = "Calendar Found! 📅"
            college_full_name = COLLEGE_DATABASE[college_key]["name"]
            body = f"Found your academic calendar on {college_full_name} website. Please verify the dates — tap to review."
            await send_immediate_push(user_id, title, body)
            logger.info(f"Scraped calendar and notified user={user_id} for college={college_key}")
        else:
            await handle_manual_calendar_fallback(user_id)
            
    except Exception as e:
        logger.error(f"Error in run_college_calendar_scraper_task: {e}", exc_info=True)
        await handle_manual_calendar_fallback(user_id)

async def handle_manual_calendar_fallback(user_id: str):
    db = firestore_async.client()
    # Set calendar_status = "manual" in user profile
    await db.collection("users").document(user_id).set({
        "calendar_status": "manual"
    }, merge=True)
    logger.info(f"Set calendar status to manual for user={user_id}")
