import os
import re
import html
import json
import fitz  # PyMuPDF
import time
import requests
import dateutil.parser
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

FRESHDESK_API_KEY = os.getenv("FRESHDESK_API_KEY")
FRESHDESK_DOMAIN = os.getenv("FRESHDESK_DOMAIN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not FRESHDESK_API_KEY or not GROQ_API_KEY:
    raise ValueError("❌ Missing API keys in the .env file!")

# Setup Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)
app = Flask(__name__)

# Freshdesk API Setup
BASE_URL = f"https://{FRESHDESK_DOMAIN}.freshdesk.com/api/v2"
AUTH = (FRESHDESK_API_KEY, "X")

# --- HELPER FUNCTIONS ---

def clean_html(raw_html):
    if not raw_html: return ""
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', raw_html)
    cleantext = html.unescape(cleantext)
    return re.sub(r'\s+', ' ', cleantext).strip()

def extract_pdf_text(attachment_url):
    try:
        resp = requests.get(attachment_url, auth=AUTH, timeout=15)
        if resp.status_code == 200:
            doc = fitz.open(stream=resp.content, filetype="pdf")
            text = "".join(page.get_text() for page in doc)
            doc.close()
            return clean_html(text) if text.strip() else ""
    except Exception as e:
        print(f"PDF Error: {e}")
    return ""

def extract_with_ai_single(chunk_text):
    """Processes a single chunk of text through Groq"""
    prompt = f"""
    You are a strict data extraction assistant. Analyze the text below.
    DO NOT calculate any math. Return ALL strings on a SINGLE LINE.

    Extract ONLY the following:
    1. "commission_value": The exact number/value (e.g., "3.5%", "$350", "5"). Look for 'Commission', 'Rate', 'Fee'.
    2. "min_lease": The exact number for the minimum duration (e.g., "3").
    3. "max_lease": The exact number for the maximum duration (e.g., "6").
    4. "min_move_in": The exact text for the minimum move-in (date or number).
    5. "max_move_in": The exact text for the maximum move-in (date or number).
    6. "start_date": The date of the specific email mentioning the Commission Value. Format YYYY-MM-DD.

    If a field is not found in this specific text chunk, return "Not Found" for that field.
    Return ONLY a valid JSON object with these exact keys: "commission_value", "min_lease", "max_lease", "min_move_in", "max_move_in", "start_date". No markdown.

    Text Chunk:
    {chunk_text}
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You output valid JSON only. You extract exact numbers. Commission values can be plain numbers without symbols."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile", 
            response_format={ "type": "json_object" }
        )
        return json.loads(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"AI Chunk Error: {e}")
        return {}

def chunk_text(text, max_length=8000):
    """Splits massive text into smaller chunks, trying to break at logical points"""
    chunks = []
    while len(text) > max_length:
        # Find the last newline or space before the max_length to avoid cutting words
        split_at = text.rfind('\n', 0, max_length)
        if split_at == -1:
            split_at = text.rfind(' ', 0, max_length)
        if split_at == -1:
            split_at = max_length
            
        chunks.append(text[:split_at])
        text = text[split_at:]
    chunks.append(text)
    return chunks

def extract_with_ai_chunked(combined_text):
    """The Chunk & Merge Engine"""
    chunks = chunk_text(combined_text, max_length=8000)
    
    # Store found values in lists to merge later
    merged_data = {
        "commission_value": [],
        "min_lease": [],
        "max_lease": [],
        "min_move_in": [],
        "max_move_in": [],
        "start_date": []
    }

    for i, chunk in enumerate(chunks):
        print(f"   Processing chunk {i+1} of {len(chunks)}...")
        chunk_result = extract_with_ai_single(chunk)
        
        # Merge findings
        for key in merged_data.keys():
            val = chunk_result.get(key, "Not Found")
            # If the AI found something and it's not a duplicate, add it to our list
            if val and val != "Not Found" and str(val) not in merged_data[key]:
                merged_data[key].append(str(val))
        
        # CRITICAL: Sleep to avoid hitting Groq's Tokens-Per-Minute (TPM) limit
        if i < len(chunks) - 1:
            time.sleep(4) 

    # Combine the lists into a single string separated by " | " for the QC comparison
    final_data = {}
    for key, val_list in merged_data.items():
        if val_list:
            final_data[key] = " | ".join(val_list)
        else:
            final_data[key] = "Not Found"
            
    return final_data

def extract_number(value):
    if not value or value == "Not Found": return None
    match = re.search(r'[\d,]+\.?\d*', str(value).replace(',', ''))
    return float(match.group()) if match else None

def standardize_date(value):
    if not value or value == "Not Found": return None
    try:
        return dateutil.parser.parse(str(value))
    except:
        return None

# --- QC LOGIC ---

def perform_qc(sheet_data, freshdesk_data):
    notes = []
    status = "✅ PASS"

    # 1. Commission Value
    sheet_val = extract_number(sheet_data.get("commission_value"))
    fd_val = extract_number(freshdesk_data.get("commission_value"))
    if sheet_val is not None and fd_val is not None:
        if abs(sheet_val - fd_val) > 0.1: 
            notes.append(f"Commission Mismatch (Sheet: {sheet_val}, FD: {fd_val})")
            status = "❌ FAIL"
    elif fd_val is None and sheet_val is not None:
        status = "⚠️ MANUAL REVIEW"
        notes.append("Commission not found in Freshdesk")

    # 2. Min Lease
    sheet_val = extract_number(sheet_data.get("min_lease"))
    fd_val = extract_number(freshdesk_data.get("min_lease"))
    if sheet_val is not None and fd_val is not None:
        if abs(sheet_val - fd_val) > 0.1:
            notes.append(f"Min Lease Mismatch (Sheet: {sheet_val}, FD: {fd_val})")
            status = "❌ FAIL"

    # 3. Max Lease
    sheet_val = extract_number(sheet_data.get("max_lease"))
    fd_val = extract_number(freshdesk_data.get("max_lease"))
    if sheet_val is not None and fd_val is not None:
        if abs(sheet_val - fd_val) > 0.1:
            notes.append(f"Max Lease Mismatch (Sheet: {sheet_val}, FD: {fd_val})")
            status = "❌ FAIL"

    # 4. Min Move In (with Timezone Tolerance)
    sheet_date = standardize_date(sheet_data.get("min_move_in"))
    fd_date = standardize_date(freshdesk_data.get("min_move_in"))
    if sheet_date and fd_date:
        if abs((sheet_date - fd_date).days) > 2:
            notes.append(f"Min Move-in Mismatch (Sheet: {sheet_date.strftime('%Y-%m-%d')}, FD: {fd_date.strftime('%Y-%m-%d')})")
            status = "❌ FAIL"

    # 5. Max Move In
    sheet_date = standardize_date(sheet_data.get("max_move_in"))
    fd_date = standardize_date(freshdesk_data.get("max_move_in"))
    if sheet_date and fd_date:
        if abs((sheet_date - fd_date).days) > 2:
            notes.append(f"Max Move-in Mismatch (Sheet: {sheet_date.strftime('%Y-%m-%d')}, FD: {fd_date.strftime('%Y-%m-%d')})")
            status = "❌ FAIL"

    # 6. Start Date
    sheet_date = standardize_date(sheet_data.get("start_date"))
    fd_date = standardize_date(freshdesk_data.get("start_date"))
    if sheet_date and fd_date:
        if abs((sheet_date - fd_date).days) > 2:
            notes.append(f"Start Date Mismatch (Sheet: {sheet_date.strftime('%Y-%m-%d')}, FD: {fd_date.strftime('%Y-%m-%d')})")
            status = "❌ FAIL"

    if not notes and status == "✅ PASS":
        notes.append("All fields match perfectly")

    return status, "; ".join(notes)

# --- FLASK ROUTES ---

@app.route('/', methods=['GET'])
def wake_up():
    return "Server is awake and ready!", 200

@app.route('/qc', methods=['POST'])
def handle_qc():
    data = request.json
    raw_urls = data.get("freshdesk_ticket_url", "")
    
    try:
        urls_list = json.loads(raw_urls)
        if not isinstance(urls_list, list):
            urls_list = [urls_list]
    except:
        urls_list = [raw_urls] if raw_urls else []

    if not urls_list:
        return jsonify({"error": "No Freshdesk links provided"}), 400

    combined_text = ""

    for ticket_url in urls_list:
        try:
            ticket_id = ticket_url.strip('/').split('/')[-1]
            int(ticket_id)
        except:
            continue

        try:
            ticket_resp = requests.get(f"{BASE_URL}/tickets/{ticket_id}", auth=AUTH, timeout=15)
            if ticket_resp.status_code != 200:
                continue
            
            ticket_data = ticket_resp.json()
            combined_text += f"\n\n=== TICKET ID: {ticket_id} ===\n"
            combined_text += f"[Main Ticket Email - Date: {ticket_data.get('created_at', '')}]\n{clean_html(ticket_data.get('description', ''))}\n\n"
            
            for att in ticket_data.get("attachments", []):
                if "pdf" in att.get("content_type", "").lower() or att.get("name", "").lower().endswith(".pdf"):
                    combined_text += f"[PDF: {att['name']}]\n{extract_pdf_text(att['attachment_url'])}\n\n"

            conv_resp = requests.get(f"{BASE_URL}/tickets/{ticket_id}/conversations", auth=AUTH, timeout=15)
            if conv_resp.status_code == 200:
                for conv in conv_resp.json():
                    combined_text += f"[Conversation Email - Date: {conv.get('created_at', '')}]\n{clean_html(conv.get('body', ''))}\n\n"
                    for att in conv.get("attachments", []):
                        if "pdf" in att.get("content_type", "").lower() or att.get("name", "").lower().endswith(".pdf"):
                            combined_text += f"[PDF: {att['name']}]\n{extract_pdf_text(att['attachment_url'])}\n\n"
                            
        except Exception as e:
            print(f"Error fetching ticket {ticket_id}: {e}")

    if not combined_text.strip():
        return jsonify({"error": "Could not fetch data from any provided Freshdesk links"}), 400

    # Call the Chunk & Merge engine instead of the old single-call
    ai_data = extract_with_ai_chunked(combined_text)

    sheet_data_for_qc = {
        "commission_value": data.get("sheet_commission_value"),
        "min_lease": data.get("sheet_min_lease"),
        "max_lease": data.get("sheet_max_lease"),
        "min_move_in": data.get("sheet_min_move_in"),
        "max_move_in": data.get("sheet_max_move_in"),
        "start_date": data.get("sheet_start_date")
    }

    qc_status, qc_notes = perform_qc(sheet_data_for_qc, ai_data)

    response_data = {
        "qc_timestamp": data.get("current_time", ""),
        "commission_id": data.get("commission_id", ""),
        "property_name": data.get("property_name", ""),
        "inventory_id": data.get("inventory_id", ""),
        "change_type": data.get("change_type", ""),
        "sheet_commission_value": data.get("sheet_commission_value", ""),
        "freshdesk_commission_value": ai_data.get("commission_value", "Not Found"),
        "sheet_min_lease": data.get("sheet_min_lease", ""),
        "freshdesk_min_lease": ai_data.get("min_lease", "Not Found"),
        "sheet_max_lease": data.get("sheet_max_lease", ""),
        "freshdesk_max_lease": ai_data.get("max_lease", "Not Found"),
        "sheet_min_move_in": data.get("sheet_min_move_in", ""),
        "freshdesk_min_move_in": ai_data.get("min_move_in", "Not Found"),
        "sheet_max_move_in": data.get("sheet_max_move_in", ""),
        "freshdesk_max_move_in": ai_data.get("max_move_in", "Not Found"),
        "sheet_start_date": data.get("sheet_start_date", ""),
        "freshdesk_start_date": ai_data.get("start_date", "Not Found"),
        "qc_status": qc_status,
        "qc_notes": qc_notes,
        "freshdesk_ticket_url": raw_urls
    }

    return jsonify(response_data), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)