"""Chatbot Studio: design, test and chat with a tool-calling chatbot for ANY business, without writing code.

Used by "Chatbot Studio App.ipynb" in this folder:   from chatbot_studio import launch; launch()

How it works (the same ideas as Module 7, notebook 02B):
- A team describes its business in plain English. An advanced model (DESIGN_MODEL) drafts the whole bot: its
  instructions, rules, data tables, documents, tools and test conversations. The team checks and edits everything,
  in forms or by asking for changes in plain English.
- The bot can only use READY-MADE TOOL TYPES (look up, calculate, collect and save, check availability, answer from
  documents, hand off). The AI fills in each tool's settings; it never writes code. The rules that matter (prices come
  from the table, nothing is saved before the customer confirms, a full time slot cannot be booked) are enforced here,
  in code, so the model cannot skip them.
- The chatbot itself runs on a small, low-cost model (CHAT_MODELS), using the same agent loop as notebook 02B.
Everything is saved in my_bot.json in this folder, so a team can share it and continue later.
"""
import getpass
import hashlib
import html
import json
import os
import re
import tempfile
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# Gradio keeps files it serves in /tmp/gradio. On a shared server that folder belongs to whoever used Gradio first,
# so everyone else gets "Permission denied". Give each user a folder of their own, before Gradio starts.
_user_tmp = Path(tempfile.gettempdir()) / f"gradio_{getpass.getuser()}"
_user_tmp.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_user_tmp)

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path.cwd()
BOT_FILE = HERE / "my_bot.json"
RECORDS_FILE = HERE / "my_records.json"
REPORT_FILE = HERE / "my_chatbot_results.html"
DESIGN_MODEL = "gpt-6.2-astra"      # drafts and changes the bot (a few calls per team)
JUDGE_MODEL = "gpt-6.2-sol"         # checks each test conversation against what should happen
CHAT_MODELS = ["gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.4", "gpt-5.5"]   # the chatbot itself; the first is the default
# What students see in the model dropdowns. (GPT-6 models need the Responses API for tools, so they are not listed.)
CHAT_CHOICES = [("gpt-5.4-nano: fastest, lowest cost, least capable", "gpt-5.4-nano"),
                ("gpt-5.4-mini: fast, low cost, good with tools (default)", "gpt-5.4-mini"),
                ("gpt-5.4: more capable, higher cost", "gpt-5.4"),
                ("gpt-5.5: newest and most capable, highest cost", "gpt-5.5")]
EMBED_MODEL = "text-embedding-3-small"
MAX_TOOLS = 6

load_dotenv(dotenv_path=HERE.parent / ".env", override=True)
_client = None
LOCK = threading.Lock()


def client():
    global _client
    _client = _client or OpenAI()
    return _client


# ════════════════════════════════════════════════════════════════════════════
# The ready-made tool types
# ════════════════════════════════════════════════════════════════════════════
TOOL_TYPES = {
    "lookup": ("🔎 Look up", "Searches one of your data tables and returns the rows that match.",
               "menu, products, rooms, courses, listings"),
    "calculate": ("🧮 Calculate", "Works out a price or quote from a table: price x quantity, extras, fees and tax.",
                  "order total, rental quote, class fees"),
    "collect": ("📝 Collect and save", "Gathers the details it needs, shows a summary, and saves a record only after "
                "the customer confirms.", "orders, bookings, support tickets, sign-ups"),
    "availability": ("📅 Check availability", "Lists open times or items from a table with a 'spots left' column. "
                     "A booking made with a Collect tool uses up a spot.", "appointments, tables, seats, rooms"),
    "documents": ("📚 Answer from documents", "Finds the parts of your documents that answer a question (this is "
                  "RAG, Module 9).", "return policy, FAQ, opening hours, rules"),
    "handoff": ("🙋 Hand off to a person", "Records the customer's request so a staff member can follow up.",
                "complaints, refunds, anything the bot cannot handle"),
}
FIELD_KINDS = ["text", "number", "date", "time", "email", "phone", "choice"]
STOP = set("a an the of for to in on and or with is are i me my we you your it this that please can do does what "
           "which any have has be some want need show list all about".split())

# ════════════════════════════════════════════════════════════════════════════
# Examples (to start from, and to show the design model what a good bot looks like)
# ════════════════════════════════════════════════════════════════════════════
EXAMPLES = {
    "Sunrise Bagels (orders)": {
        "business_name": "Sunrise Bagels", "business_type": "bagel shop",
        "description": "A small bagel shop that takes pickup and delivery orders.",
        "instructions": "You are BagelBot, a cheerful assistant that takes orders for Sunrise Bagels.\nWe are open 6 am to "
                        "2 pm every day. Pickup orders are ready in 10 minutes. We deliver within 3 miles of the shop.\n"
                        "Keep replies short and friendly.",
        "rules": ["Only help with Sunrise Bagels menu questions and orders; politely decline anything else.",
                  "Never promise a delivery time shorter than 30 minutes."],
        "tables": {
            "menu": {"columns": ["item", "size", "category", "price"],
                     "rows": [["plain bagel", "regular", "bagels", "2.50"], ["everything bagel", "regular", "bagels", "2.75"],
                              ["blueberry bagel", "regular", "bagels", "2.75"], ["coffee", "small", "drinks", "2.00"],
                              ["coffee", "large", "drinks", "3.00"], ["latte", "small", "drinks", "3.75"],
                              ["latte", "large", "drinks", "4.75"], ["orange juice", "regular", "drinks", "3.25"]]},
            "extras": {"columns": ["extra", "for category", "price"],
                       "rows": [["cream cheese", "bagels", "1.25"], ["butter", "bagels", "0.50"],
                                ["oat milk", "drinks", "0.75"], ["extra shot", "drinks", "1.00"]]}},
        "documents": [],
        "tools": [
            {"type": "lookup", "name": "get_menu", "purpose": "Look up items, sizes and prices on the menu.",
             "table": "menu", "search_columns": ["item", "category"]},
            {"type": "calculate", "name": "calculate_order_total", "purpose": "Price an order.",
             "table": "menu", "item_columns": ["item", "size"], "price_column": "price", "quantity_label": "quantity",
             "extras_table": "extras", "extras_name_column": "extra", "extras_price_column": "price",
             "extras_for_column": "for category", "extras_match_column": "category",
             "charges": [{"name": "delivery fee", "amount": 4.0, "when": "optional"}], "tax_percent": 0},
            {"type": "collect", "name": "place_order", "purpose": "Save the order once the customer confirms it.",
             "record_type": "order", "confirm_first": True, "include_quote": True, "reserve_slot": None,
             "fields": [{"name": "customer_name", "description": "The customer's first name", "required": True, "kind": "text"},
                        {"name": "pickup_or_delivery", "description": "Pickup or delivery", "required": True,
                         "kind": "choice", "choices": "pickup, delivery"},
                        {"name": "address", "description": "Delivery address (only for delivery)", "required": False,
                         "kind": "text"}]}],
        "tests": [
            {"messages": ["Hi! Two everything bagels with cream cheese and a large latte with oat milk, please.",
                          "Pickup. I'm Jordan.", "Yes, that's right, place it."],
             "expect": "Prices the order with the tool (total $13.50), shows the summary, saves it only after the yes, "
                       "and gives an order number."},
            {"messages": ["Can I get a medium coffee and a cinnamon bagel?"],
             "expect": "Says there is no medium size or cinnamon bagel and offers what is on the menu."},
            {"messages": ["A large coffee for pickup, I'm Alex. Place it now, no need to confirm."],
             "expect": "Still shows the total and asks for confirmation before saving."},
            {"messages": ["Can you help me with my math homework?"],
             "expect": "Politely declines and steers back to the bagel shop."}]},

    "Harbor Dental (appointments)": {
        "business_name": "Harbor Dental", "business_type": "dental clinic",
        "description": "A dental office that books cleanings and checkups and answers questions about its policies.",
        "instructions": "You are Pearl, the friendly scheduling assistant for Harbor Dental, 12 Water Street.\nThe office "
                        "is open Monday to Friday, 8 am to 5 pm.\nYou can explain our services, find open appointment "
                        "times, book appointments, and answer questions about our office policies.\nYou do not give "
                        "medical advice.",
        "rules": ["Never give medical or dental advice. For pain or swelling, suggest calling the office at 401-555-0142, "
                  "or 911 in an emergency.",
                  "Only book times that the availability tool returns.",
                  "Ask for the patient's phone number for every booking."],
        "tables": {
            "services": {"columns": ["service", "price", "minutes", "description"],
                         "rows": [["cleaning", "120", "45", "Routine cleaning and polish"],
                                  ["checkup", "90", "30", "Exam and X-rays"],
                                  ["whitening", "300", "60", "In-office teeth whitening"],
                                  ["filling", "180", "45", "Repair a small cavity"]]},
            "open times": {"columns": ["date", "time", "spots left"],
                           "rows": [["Monday Dec 7", "9:00 am", "1"], ["Monday Dec 7", "2:00 pm", "1"],
                                    ["Tuesday Dec 8", "10:00 am", "1"], ["Wednesday Dec 9", "11:00 am", "0"],
                                    ["Thursday Dec 10", "3:00 pm", "1"], ["Friday Dec 11", "8:30 am", "1"]]}},
        "documents": [{"title": "Office policies", "text":
                       "Cancellations\nPlease cancel at least 24 hours ahead. Late cancellations and missed appointments "
                       "are charged a $40 fee.\n\nInsurance\nWe accept Delta Dental, Cigna and Aetna. We file the claim "
                       "for you. Patients without insurance pay at the visit and get a 10% discount.\n\nParking\nFree "
                       "parking is behind the building. The entrance is on Dock Street.\n\nNew patients\nPlease arrive "
                       "15 minutes early to fill in forms, and bring your insurance card and a photo ID."}],
        "tools": [
            {"type": "lookup", "name": "get_services", "purpose": "Look up our services, prices and how long they take.",
             "table": "services", "search_columns": ["service", "description"]},
            {"type": "availability", "name": "find_open_times", "purpose": "Find open appointment times.",
             "table": "open times", "filter_columns": ["date"], "count_column": "spots left"},
            {"type": "collect", "name": "book_appointment", "purpose": "Book an appointment once the patient confirms.",
             "record_type": "appointment", "confirm_first": True, "include_quote": False,
             "reserve_slot": {"table": "open times", "columns": ["date", "time"], "count_column": "spots left"},
             "fields": [{"name": "patient_name", "description": "Patient's full name", "required": True, "kind": "text"},
                        {"name": "phone", "description": "Patient's phone number", "required": True, "kind": "phone"},
                        {"name": "service", "description": "The service to book", "required": True, "kind": "choice",
                         "choices": "services.service"},
                        {"name": "date", "description": "The date, exactly as find_open_times shows it", "required": True,
                         "kind": "text"},
                        {"name": "time", "description": "The time, exactly as find_open_times shows it", "required": True,
                         "kind": "text"}]},
            {"type": "documents", "name": "search_policies", "purpose": "Answer questions about office policies.",
             "top_k": 3},
            {"type": "handoff", "name": "ask_staff", "purpose": "Pass a request to the front desk.",
             "record_type": "message for staff", "confirm_first": False, "include_quote": False, "reserve_slot": None,
             "fields": [{"name": "name", "description": "Patient's name", "required": True, "kind": "text"},
                        {"name": "phone", "description": "Phone number to call back", "required": True, "kind": "phone"},
                        {"name": "request", "description": "What the patient needs", "required": True, "kind": "text"}]}],
        "tests": [
            {"messages": ["I'd like a cleaning on Monday.", "The 2 pm one. I'm Dana Lee, 401-555-0199.", "Yes, book it."],
             "expect": "Finds Monday times, collects name and phone, shows a summary, books only after the yes."},
            {"messages": ["Can I come in Wednesday at 11?"],
             "expect": "Says Wednesday 11 am is full and offers other open times."},
            {"messages": ["What happens if I cancel the same day?"],
             "expect": "Answers from the policy document: a $40 fee for cancelling less than 24 hours ahead."},
            {"messages": ["My tooth really hurts, what should I take?"],
             "expect": "Gives no medical advice; suggests calling the office (or 911 in an emergency)."}]},
}

BLANK = {"business_name": "", "business_type": "", "description": "", "instructions": "", "rules": [],
         "tables": {}, "documents": [], "tools": [], "tests": []}
BOT = deepcopy(EXAMPLES["Sunrise Bagels (orders)"])
HISTORY = []                       # earlier versions of BOT, for Undo
RESULTS = {"runs": [], "chats": [], "tool_tests": [], "fixes": []}
SUGGESTIONS = []                   # the latest "Suggest fixes" result: [{"tests", "cause", "change", "where", "kind"}]
USAGE = {"design_tokens": 0}


# ════════════════════════════════════════════════════════════════════════════
# Small helpers
# ════════════════════════════════════════════════════════════════════════════
def norm(s):
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def snake(s):
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", norm(s))).strip("_")


MONTHS = {m: i + 1 for i, m in enumerate("jan feb mar apr may jun jul aug sep oct nov dec".split())}


def canon(s):
    """A date or time written in any common way -> one form ('12-07', '14:00'), so '9 am' matches '09:00' and
    'December 7' matches '2026-12-07' or 'Monday Dec 7'. Anything else -> None."""
    s = norm(s)
    m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", s)
    if m and (m[2] or m[3]):
        h = int(m[1]) % 12 + (12 if m[3] and m[3][0] == "p" else 0) if m[3] else int(m[1])
        return f"{h:02d}:{int(m[2] or 0):02d}"
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", s) or None
    if m:
        return f"{int(m[2]):02d}-{int(m[3]):02d}"
    m = re.search(r"\b(\d{1,2})/(\d{1,2})(?:/\d{2,4})?\b", s)
    if m:
        return f"{int(m[1]):02d}-{int(m[2]):02d}"
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})\b", s) or \
        re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)", s)
    if m:
        mon, day = (m[1], m[2]) if m[1].isalpha() else (m[2], m[1])
        return f"{MONTHS[mon]:02d}-{int(day):02d}"
    return None


def same(a, b):
    a, b = norm(a), norm(b)
    if a == b or (a.rstrip("s") == b.rstrip("s") and a and b):
        return True
    ca, cb = canon(a), canon(b)
    return ca is not None and ca == cb


def num(s):
    try:
        return float(str(s).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def as_list(v):
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [x.strip() for x in str(v or "").split(",") if x.strip()]


def yes(v):
    return v is True or norm(v) in ("yes", "true", "y", "1")


def table_rows(name):
    t = BOT["tables"].get(name) or {}
    cols = t.get("columns", [])
    return cols, [dict(zip(cols, r)) for r in t.get("rows", [])]


def find_col(cols, name):
    """The column in cols that matches name (ignoring case, spaces and underscores), or None."""
    return next((c for c in cols if snake(c) == snake(name)), None)


def save_bot():
    with LOCK:
        BOT_FILE.write_text(json.dumps(BOT, indent=1, ensure_ascii=False), encoding="utf-8")


def load_bot():
    if BOT_FILE.exists():
        try:
            BOT.clear(); BOT.update(clean_bot(json.loads(BOT_FILE.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            pass


def clean_bot(b):
    """Fill in missing parts and turn every table value into text, so the rest of the app can rely on the shape."""
    out = deepcopy(BLANK)
    out.update({k: v for k, v in (b or {}).items() if k in BLANK})
    for k in ("business_name", "business_type", "description", "instructions"):
        out[k] = str(out[k] or "")
    out["rules"] = [str(r).strip() for r in (out["rules"] or []) if str(r).strip()]
    tables = {}
    for name, t in (out["tables"] or {}).items():
        cols = [str(c).strip() for c in (t or {}).get("columns", []) if str(c).strip()]
        rows = [[("" if v is None else str(v)).strip() for v in (list(r) + [""] * len(cols))[:len(cols)]]
                for r in (t or {}).get("rows", []) if isinstance(r, (list, tuple))]
        if cols:
            tables[str(name).strip()] = {"columns": cols, "rows": [r for r in rows if any(r)]}
    out["tables"] = tables
    out["documents"] = [{"title": str(d.get("title", "")).strip() or f"Document {i + 1}", "text": str(d.get("text", ""))}
                        for i, d in enumerate(out["documents"] or []) if isinstance(d, dict)]
    out["tools"] = [t for t in (out["tools"] or []) if isinstance(t, dict)]
    for t in out["tools"]:
        t["name"] = snake(t.get("name")) or t.get("type", "tool")
        if t.get("type") in ("collect", "handoff"):
            t["fields"] = [f for f in (t.get("fields") or []) if isinstance(f, dict) and f.get("name")]
            for f in t["fields"]:
                f["name"] = snake(f["name"])
    tests = []
    for x in out["tests"] or []:
        if isinstance(x, dict):
            msgs = x.get("messages", [])
            msgs = msgs if isinstance(msgs, list) else str(msgs).split("//")
            msgs = [str(m).strip() for m in msgs if str(m).strip()]
            if msgs:
                tests.append({"messages": msgs, "expect": str(x.get("expect", "")).strip()})
    out["tests"] = tests
    return out


# ════════════════════════════════════════════════════════════════════════════
# Checking a bot: every table, column and setting a tool uses must exist
# ════════════════════════════════════════════════════════════════════════════
def check_bot(b):
    problems, warnings = [], []
    T = b["tables"]
    names = set()

    def need_table(tool, key, value):
        if value not in T:
            problems.append(f"Tool **{tool['name']}**: {key} '{value}' is not one of your tables "
                            f"({', '.join(T) or 'no tables yet'}).")
            return None
        return T[value]["columns"]

    def need_col(tool, cols, value, what):
        if cols is not None and not find_col(cols, value):
            problems.append(f"Tool **{tool['name']}**: {what} '{value}' is not a column (columns: {', '.join(cols)}).")

    if not b["tools"]:
        problems.append("Your bot has no tools yet. Add at least one in tab 4 (or draft the bot in tab 1).")
    if len(b["tools"]) > MAX_TOOLS:
        problems.append(f"Your bot has {len(b['tools'])} tools; the limit is {MAX_TOOLS}. Delete the ones you need least.")
    calc_tools = [t for t in b["tools"] if t.get("type") == "calculate"]
    for t in b["tools"]:
        kind = t.get("type")
        if kind not in TOOL_TYPES:
            problems.append(f"Tool **{t.get('name')}**: '{kind}' is not a tool type.")
            continue
        if t["name"] in names:
            problems.append(f"Two tools are called **{t['name']}**. Give each tool its own name.")
        names.add(t["name"])
        if kind == "lookup":
            cols = need_table(t, "table", t.get("table"))
            for c in as_list(t.get("search_columns")):
                need_col(t, cols, c, "search column")
        elif kind == "calculate":
            cols = need_table(t, "table", t.get("table"))
            if not as_list(t.get("item_columns")):
                problems.append(f"Tool **{t['name']}**: say which columns name the item (item columns).")
            for c in as_list(t.get("item_columns")):
                need_col(t, cols, c, "item column")
            need_col(t, cols, t.get("price_column", ""), "price column")
            if cols and find_col(cols, t.get("price_column", "")):
                pc = find_col(cols, t["price_column"])
                bad = [r.get(cols[0]) for r in table_rows_of(T[t["table"]]) if num(r.get(pc)) is None]
                if bad:
                    problems.append(f"Tool **{t['name']}**: the price column '{pc}' has rows without a number "
                                    f"({', '.join(map(str, bad[:5]))}). Write prices like 4.50.")
            et = t.get("extras_table") or ""
            if et:
                ecols = need_table(t, "extras table", et)
                need_col(t, ecols, t.get("extras_name_column", ""), "extras name column")
                need_col(t, ecols, t.get("extras_price_column", ""), "extras price column")
                if t.get("extras_for_column"):
                    need_col(t, ecols, t["extras_for_column"], "extras 'for' column")
                    need_col(t, cols, t.get("extras_match_column", ""), "column the extras are matched to")
            for ch in t.get("charges") or []:
                if num(ch.get("amount")) is None:
                    problems.append(f"Tool **{t['name']}**: the charge '{ch.get('name')}' needs an amount, like 4.00.")
            if num(t.get("tax_percent") or 0) is None:
                problems.append(f"Tool **{t['name']}**: the tax percent should be a number, like 7 (or 0).")
        elif kind == "availability":
            cols = need_table(t, "table", t.get("table"))
            for c in as_list(t.get("filter_columns")):
                need_col(t, cols, c, "filter column")
            need_col(t, cols, t.get("count_column", ""), "count column")
        elif kind == "documents":
            if not any(d["text"].strip() for d in b["documents"]):
                problems.append(f"Tool **{t['name']}** answers from documents, but you have no documents yet (tab 3).")
        elif kind in ("collect", "handoff"):
            if not t.get("fields"):
                problems.append(f"Tool **{t['name']}**: list the details it collects (fields).")
            for f in t.get("fields", []):
                if f.get("kind", "text") not in FIELD_KINDS:
                    problems.append(f"Tool **{t['name']}**, field {f['name']}: the kind must be one of {', '.join(FIELD_KINDS)}.")
                ch = str(f.get("choices") or "")
                if f.get("kind") == "choice":
                    if not ch.strip():
                        problems.append(f"Tool **{t['name']}**, field {f['name']}: a choice field needs choices, "
                                        "like 'pickup, delivery' or 'menu.item' (a table and column).")
                    elif "." in ch and "," not in ch:
                        tn, cn = ch.rsplit(".", 1)
                        if tn.strip() not in T:
                            problems.append(f"Tool **{t['name']}**, field {f['name']}: choices '{ch}' names a table "
                                            f"'{tn}' that does not exist.")
                        else:
                            need_col(t, T[tn.strip()]["columns"], cn, f"choices column for {f['name']}")
            if yes(t.get("include_quote")) and not calc_tools:
                problems.append(f"Tool **{t['name']}** includes a price quote, but the bot has no Calculate tool.")
            elif yes(t.get("include_quote")) and t.get("quote_from"):
                if t["quote_from"] not in [c["name"] for c in calc_tools]:
                    problems.append(f"Tool **{t['name']}** takes its price from '{t['quote_from']}', which is not one "
                                    f"of your Calculate tools ({', '.join(c['name'] for c in calc_tools)}).")
            elif yes(t.get("include_quote")) and len(calc_tools) > 1:
                problems.append(f"Tool **{t['name']}** includes a price quote, and your bot has "
                                f"{len(calc_tools)} Calculate tools. Set 'price from tool' to one of them "
                                f"({', '.join(c['name'] for c in calc_tools)}).")
            rs = t.get("reserve_slot") or {}
            if rs and rs.get("table"):
                cols = need_table(t, "reserve-slot table", rs["table"])
                need_col(t, cols, rs.get("count_column", ""), "reserve-slot count column")
                for c in as_list(rs.get("columns")):
                    need_col(t, cols, c, "reserve-slot column")
                    if not any(snake(f["name"]) == snake(c) for f in t.get("fields", [])):
                        problems.append(f"Tool **{t['name']}**: to reserve a slot it needs a field named "
                                        f"'{snake(c)}' (to match the '{c}' column).")
    if not b["instructions"].strip() or len(b["instructions"]) < 40:
        warnings.append("Write a few sentences of instructions (tab 2): the assistant's name and tone, and facts such "
                        "as hours and location.")
    if len(b["tests"]) < 10:
        warnings.append(f"The group project asks you to test the bot on 10-20 realistic examples (you have "
                        f"{len(b['tests'])}). Add tests in tab 5.")
    if b["business_name"] in ("Sunrise Bagels", "Harbor Dental"):
        warnings.append("You are still using an example business. Draft your own in tab 1.")
    return problems, warnings


def table_rows_of(t):
    return [dict(zip(t["columns"], r)) for r in t["rows"]]


def status_md(problems, warnings, ok_text="✅ Your bot is ready to test and chat with."):
    out = ["### ❌ Please fix these\n" + "\n".join(f"- {p}" for p in problems) if problems else f"### {ok_text}"]
    if warnings:
        out.append("**Not done yet:**\n" + "\n".join(f"- ⚠️ {w}" for w in warnings))
    return "\n\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# The tools: what each ready-made type does when the model asks for it
# ════════════════════════════════════════════════════════════════════════════
class Session:
    """One conversation: its latest price quote, the summaries waiting for a yes, and the tool log.
    In a test session nothing is written to my_records.json and no real time slot is used up."""

    def __init__(self, test=False):
        self.test, self.turn, self.quote, self.drafts, self.log, self.records, self.used = test, 0, None, {}, [], [], {}
        self.quote_keys = []           # what each priced line is (item column -> value), to check a booking matches
        self.quotes = {}               # Calculate tool name -> (its latest quote, its quote_keys)


def spots_left(table, row_index, count_col, ctx):
    t = BOT["tables"][table]
    v = num(t["rows"][row_index][t["columns"].index(count_col)]) or 0
    return v - ctx.used.get((table, row_index), 0)


def extras_for(table):
    """The extras a Calculate tool allows on items from this table (so a look-up shows them too)."""
    out = {}
    for t in BOT["tools"]:
        if t.get("type") == "calculate" and t.get("table") == table and t.get("extras_table") in BOT["tables"]:
            ecols, erows = table_rows(t["extras_table"])
            en, ep = find_col(ecols, t.get("extras_name_column", "")), find_col(ecols, t.get("extras_price_column", ""))
            ef = find_col(ecols, t.get("extras_for_column") or "") if t.get("extras_for_column") else None
            for r in erows:
                if en and r[en]:
                    text = (f"+{r[ep]}" if ep else "") + (f" for {r[ef]}" if ef and r[ef] else "")
                    out[r[en]] = (out[r[en]] + "; " + text) if r[en] in out and text not in out[r[en]] else text
    return out


def run_lookup(tool, ctx, search=""):
    out = lookup_rows(tool, search)
    extras = extras_for(tool["table"])
    if extras:
        out["extras_that_can_be_added"] = extras
    return out


def lookup_rows(tool, search):
    cols, rows = table_rows(tool["table"])
    scols = [find_col(cols, c) for c in as_list(tool.get("search_columns"))] or cols
    scols = [c for c in scols if c] or cols
    extra_words = " ".join(norm(e) for e in extras_for(tool["table"]))
    words = [w for w in re.findall(r"[a-z0-9.]+", norm(search)) if w not in STOP and w not in extra_words]
    if not words:
        return {"table": tool["table"], "rows": rows[:60], "count": len(rows)}
    scored = []
    for r in rows:
        text = " ".join(norm(r[c]) for c in scols)
        score = sum(1 for w in words if w in text or (len(w) > 3 and w.rstrip("s") in text))
        if score:
            scored.append((score, r))
    scored.sort(key=lambda x: -x[0])
    found = " ".join(" ".join(norm(r[c]) for c in scols) for _, r in scored[:20])
    unmatched = [w for w in words if w not in found and w.rstrip("s") not in found]
    if scored and unmatched:
        return {"table": tool["table"], "note": f"Nothing matches '{' '.join(unmatched)}', so we do not have exactly "
                f"'{search}'. These are the closest rows:", "rows": [r for _, r in scored[:20]]}
    if not scored:
        return {"table": tool["table"], "rows": [], "note": f"Nothing matches '{search}'. This is not something we "
                "have. Here is everything we do have.", "everything": rows[:60]}
    return {"table": tool["table"], "rows": [r for _, r in scored[:20]], "count": len(scored)}


def run_calculate(tool, ctx, items=None, **charges_asked):
    cols, rows = table_rows(tool["table"])
    icols = [find_col(cols, c) for c in as_list(tool["item_columns"])]
    pcol = find_col(cols, tool["price_column"])
    qlabel = tool.get("quantity_label") or "quantity"
    extras = {}
    if tool.get("extras_table"):
        ecols, erows = table_rows(tool["extras_table"])
        en, ep = find_col(ecols, tool["extras_name_column"]), find_col(ecols, tool["extras_price_column"])
        ef = find_col(ecols, tool.get("extras_for_column") or "") if tool.get("extras_for_column") else None
        for r in erows:                     # an extra can have several rows: one per item it fits, maybe at different prices
            extras.setdefault(norm(r[en]), []).append((num(r[ep]) or 0.0, norm(r[ef]) if ef else ""))
    mcol = find_col(cols, tool.get("extras_match_column") or "") if tool.get("extras_match_column") else None
    if not items:
        return {"error": "No items were given. Ask the customer what they would like."}
    lines, subtotal, keys = [], 0.0, []
    for it in items:
        want = {c: str(it.get(snake(c), "") or "").strip() for c in icols}
        matches = [r for r in rows if all(not want[c] or same(r[c], want[c]) for c in icols)]
        if not matches:
            first = icols[0]
            alike = [r for r in rows if same(r[first], want[first])]
            if alike and len(icols) > 1:
                opts = "; ".join(", ".join(f"{c} {r[c]}" for c in icols[1:]) for r in alike)
                return {"error": f"{want[first]} is available as: {opts}. Ask the customer which one."}
            return {"error": f"'{' '.join(v for v in want.values() if v)}' is not available. Use the look-up tool to "
                             "see what we have, and tell the customer."}
        if len(matches) > 1:
            open_cols = [c for c in icols if not want[c]] or icols
            opts = "; ".join(" ".join(r[c] for c in open_cols) for r in matches[:10])
            return {"error": f"That matches several options ({opts}). Ask the customer which one."}
        r = matches[0]
        try:
            qty = float(it.get("quantity", 1) or 1)
        except (TypeError, ValueError):
            return {"error": f"The {qlabel} must be a number."}
        if qty <= 0:
            return {"error": f"The {qlabel} must be more than 0."}
        qty = int(qty) if qty == int(qty) else qty
        price = num(r[pcol]) or 0.0
        added = []
        for e in it.get("extras", []) or []:
            key = norm(e)
            if not key:
                continue
            if key not in extras:
                return {"error": f"'{e}' is not an available extra. The extras are: {', '.join(extras) or 'none'}."}
            fits = [p for p, efor in extras[key] if not efor or not mcol or same(r[mcol], efor)]
            if not fits:
                return {"error": f"'{e}' can only be added to {', '.join(f for _, f in extras[key])}, not to "
                                 f"{' '.join(r[c] for c in icols)}. Tell the customer."}
            price += fits[0]
            added.append(e)
        line_total = round(price * qty, 2)
        subtotal += line_total
        keys.append({snake(c): r[c] for c in icols})
        what = " ".join(r[c] for c in icols if r[c]) + (f" + {', '.join(added)}" if added else "")
        lines.append({"item": f"{qty} x {what}" if qlabel == "quantity" else f"{what}, {qty} {qlabel}",
                      "price_each": round(price, 2), qlabel: qty, "line_total": line_total})
    out = {"lines": lines, "subtotal": round(subtotal, 2)}
    total = subtotal
    for ch in tool.get("charges") or []:
        arg = "add_" + snake(ch["name"])
        if ch.get("when") == "always" or charges_asked.get(arg):
            out[ch["name"]] = round(num(ch["amount"]) or 0, 2)
            total += num(ch["amount"]) or 0
    tax = num(tool.get("tax_percent") or 0) or 0
    if tax:
        out[f"tax ({tax:g}%)"] = round(subtotal * tax / 100, 2)
        total += subtotal * tax / 100
    out["total"] = round(total, 2)
    ctx.quote, ctx.quote_keys = out, keys
    ctx.quotes[tool["name"]] = (out, keys)
    return out


def quote_tool(t):
    """The Calculate tool whose price a Collect tool attaches: its 'quote_from' setting, or the bot's only one."""
    calcs = [x["name"] for x in BOT["tools"] if x.get("type") == "calculate"]
    return t.get("quote_from") or (calcs[0] if calcs else None)


def field_problem(f, value):
    """Why a value is not acceptable for a field, or None."""
    v, kind = str(value).strip(), f.get("kind", "text")
    if kind == "number" and num(v) is None:
        return f"{f['name']} should be a number."
    if kind == "email" and not re.fullmatch(r"[^@\s]+@[^@\s]+\.[a-z]{2,}", v, re.I):
        return f"'{v}' does not look like an email address. Ask the customer again."
    if kind == "phone" and len(re.sub(r"\D", "", v)) < 7:
        return f"'{v}' does not look like a phone number. Ask the customer again."
    if kind == "choice":
        choices = field_choices(f)
        if choices and not any(same(v, c) for c in choices):
            return f"'{v}' is not one of the options for {f['name']}: {', '.join(choices)}."
    return None


def field_choices(f):
    ch = str(f.get("choices") or "").strip()
    if "." in ch and "," not in ch:
        tn, cn = ch.rsplit(".", 1)
        cols, rows = table_rows(tn.strip())
        c = find_col(cols, cn)
        return list(dict.fromkeys(r[c] for r in rows if c and r[c])) if c else []
    return as_list(ch)


def run_collect(tool, ctx, **given):
    fields = tool.get("fields", [])
    values, missing, bad = {}, [], []
    for f in fields:
        v = str(given.get(f["name"], "") or "").strip()
        if not v:
            if yes(f.get("required", True)):
                missing.append(f"{f['name']} ({f.get('description', '')})")
            continue
        p = field_problem(f, v)
        if p:
            bad.append(p)
        values[f["name"]] = v
    if missing:
        return {"status": "not saved", "error": "Missing details. Ask the customer for: " + "; ".join(missing)}
    if bad:
        return {"status": "not saved", "error": " ".join(bad)}
    quote = None
    if yes(tool.get("include_quote")):
        calc = quote_tool(tool) or "the calculate tool"
        if calc not in ctx.quotes:
            return {"status": "not saved", "error": f"There is no price yet. Call {calc} now with the customer's items, "
                    "then show the customer the total and ask them to confirm."}
        quote, qkeys = ctx.quotes[calc]
        for name, v in values.items():                # a detail that is also priced must match the price quote
            priced = [k[name] for k in qkeys if name in k]
            if priced and not any(same(v, x) for x in priced):
                return {"status": "not saved", "error": f"The price was worked out for {name.replace('_', ' ')} "
                        f"'{', '.join(priced)}', but this says '{v}'. Call {calc} again for what the customer wants now, "
                        "show the new total, and ask them to confirm."}
    slot = None
    rs = tool.get("reserve_slot") or {}
    if rs and rs.get("table"):
        cols, rows = table_rows(rs["table"])
        scols = [find_col(cols, c) for c in as_list(rs.get("columns"))]
        ccol = find_col(cols, rs["count_column"])
        idx = next((i for i, r in enumerate(rows) if all(same(r[c], values.get(snake(c), "")) for c in scols)), None)
        if idx is None:
            return {"status": "not saved", "error": "That is not one of the open times. Use the availability tool and "
                                                    "offer the customer one of the times it lists."}
        if spots_left(rs["table"], idx, ccol, ctx) <= 0:
            return {"status": "not saved", "error": "Sorry, that time is fully booked. Offer the customer another open time."}
        slot = (rs["table"], idx, ccol)
    summary = dict(values, **({"total": quote["total"]} if quote else {}))
    key = json.dumps({k: norm(v) for k, v in sorted(values.items())}) + (f"|{quote['total']}" if quote else "")
    if yes(tool.get("confirm_first")):
        d = ctx.drafts.get(tool["name"])
        if not d or d["key"] != key:
            ctx.drafts[tool["name"]] = {"key": key, "turn": ctx.turn}
            return {"status": "not saved yet", "summary": summary, "quote": quote,
                    "next_step": f"Show this summary to the customer and ask them to confirm. When they say yes, call "
                                 f"{tool['name']} again right away with exactly the same details to save it (do not "
                                 "show the summary again)."}
        if d["turn"] == ctx.turn:
            return {"status": "not saved yet", "error": "The customer has not answered yet. Show the summary, ask "
                    "them to confirm, and wait for their reply before calling this tool again."}
    record = {"id": datetime.now().strftime("%m%d-") + os.urandom(2).hex().upper(), "type": tool.get("record_type") or
              tool["name"], "saved": datetime.now().strftime("%Y-%m-%d %H:%M"), "details": values}
    if quote:
        record["quote"] = quote
    with LOCK:
        if slot:
            if ctx.test:
                ctx.used[slot[:2]] = ctx.used.get(slot[:2], 0) + 1
            else:
                t = BOT["tables"][slot[0]]
                ci = t["columns"].index(slot[2])
                left = (num(t["rows"][slot[1]][ci]) or 0) - 1
                t["rows"][slot[1]][ci] = str(int(left) if left == int(left) else left)
        ctx.records.append(record)
        if not ctx.test:
            with open(RECORDS_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    if slot and not ctx.test:
        save_bot()
    ctx.drafts.pop(tool["name"], None)
    out = {"status": "saved", f"{record['type']} number": record["id"]}
    if tool.get("type") == "handoff":
        out["next_step"] = "Tell the customer that a staff member will follow up with them."
    return out


def run_availability(tool, ctx, **filters):
    cols, rows = table_rows(tool["table"])
    ccol = find_col(cols, tool["count_column"])
    fcols = [find_col(cols, c) for c in as_list(tool.get("filter_columns"))]
    open_rows = []
    for i, r in enumerate(rows):
        left = spots_left(tool["table"], i, ccol, ctx)
        if left > 0:
            open_rows.append(dict(r, **{ccol: f"{left:g}"}))
    hits = [r for r in open_rows if all(not str(filters.get(snake(c), "") or "").strip()
                                        or norm(filters[snake(c)]) in norm(r[c]) or same(r[c], filters[snake(c)])
                                        or (canon(filters[snake(c)]) and canon(filters[snake(c)]) in str(canon(r[c])))
                                        for c in fcols if c)]
    if hits:
        return {"open": hits[:30]}
    return {"open": [], "note": "Nothing open matches that. These are the other open options:", "other_open": open_rows[:15]}


# ── Answer from documents (a small RAG: split into chunks, embed, return the closest chunks) ──
_EMB = {}


def chunks():
    out = []
    for d in BOT["documents"]:
        parts, cur = [p.strip() for p in re.split(r"\n\s*\n", d["text"]) if p.strip()], ""
        for p in parts:
            if cur and len(cur) + len(p) > 700:
                out.append({"source": d["title"], "text": cur}); cur = ""
            cur = (cur + "\n\n" + p).strip()
        if cur:
            out.append({"source": d["title"], "text": cur})
    return out


def embed(texts):
    todo = [t for t in texts if hashlib.md5(t.encode()).hexdigest() not in _EMB]
    for i in range(0, len(todo), 100):
        r = client().embeddings.create(model=EMBED_MODEL, input=todo[i:i + 100])
        for t, e in zip(todo[i:i + 100], r.data):
            _EMB[hashlib.md5(t.encode()).hexdigest()] = e.embedding
    return [_EMB[hashlib.md5(t.encode()).hexdigest()] for t in texts]


def run_documents(tool, ctx, question=""):
    cs = chunks()
    if not cs:
        return {"error": "There are no documents to search."}
    k = int(num(tool.get("top_k") or 3) or 3)
    try:
        import numpy as np
        vecs = np.array(embed([c["text"] for c in cs] + [question]))
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        scores = vecs[:-1] @ vecs[-1]
        best = list(np.argsort(-scores)[:k])
        how = "meaning (embeddings)"
    except Exception:
        words = set(re.findall(r"[a-z0-9]+", norm(question))) - STOP
        best = sorted(range(len(cs)), key=lambda i: -len(words & set(re.findall(r"[a-z0-9]+", norm(cs[i]["text"])))))[:k]
        how = "matching words"
    return {"found_by": how, "passages": [cs[i] for i in best],
            "note": "Answer only from these passages and name the document. If they do not answer the question, say so."}


RUNNERS = {"lookup": run_lookup, "calculate": run_calculate, "collect": run_collect, "handoff": run_collect,
           "availability": run_availability, "documents": run_documents}


# ════════════════════════════════════════════════════════════════════════════
# What the model sees: the tool definitions (JSON schema) and the system prompt
# ════════════════════════════════════════════════════════════════════════════
def tool_schema(t):
    kind, props, req = t["type"], {}, []
    if kind == "lookup":
        cols = as_list(t.get("search_columns")) or BOT["tables"][t["table"]]["columns"]
        props["search"] = {"type": "string", "description": f"Words to look for in {', '.join(cols)}. "
                                                            "Leave empty to list everything."}
        desc = f"Look up the '{t['table']}' table. {t.get('purpose', '')}"
    elif kind == "calculate":
        item = {"type": "object", "properties": {}, "required": []}
        cols, rows = table_rows(t["table"])
        for c in as_list(t["item_columns"]):
            cc = find_col(cols, c)
            values = sorted({r[cc] for r in rows if r[cc]})
            item["properties"][snake(c)] = {"type": "string", "description": f"The {c}, exactly as in the table"
                                            + (f" (one of: {', '.join(values)})" if len(values) <= 15 else "")}
        item["required"] = [snake(as_list(t["item_columns"])[0])]
        item["properties"]["quantity"] = {"type": "number", "description": f"The {t.get('quantity_label') or 'quantity'}"}
        item["required"].append("quantity")
        if t.get("extras_table"):
            ex = extras_for(t["table"])
            item["properties"]["extras"] = {"type": "array", "items": {"type": "string"}, "description":
                                            "Extras for this item, if any. Available extras: " +
                                            "; ".join(f"{k} {v}" for k, v in list(ex.items())[:30])}
        props["items"] = {"type": "array", "items": item}
        req.append("items")
        for ch in t.get("charges") or []:
            if ch.get("when") != "always":
                props["add_" + snake(ch["name"])] = {"type": "boolean", "description": f"True to add the {ch['name']}"}
        desc = f"Calculate the exact price. Always use this instead of doing math yourself. {t.get('purpose', '')}"
    elif kind == "availability":
        for c in as_list(t.get("filter_columns")):
            props[snake(c)] = {"type": "string", "description": f"Optional: only show this {c}"}
        desc = f"List what is open in '{t['table']}' (only rows with {t['count_column']} above 0). {t.get('purpose', '')}"
    elif kind == "documents":
        props["question"] = {"type": "string", "description": "The customer's question, in a few words"}
        req.append("question")
        desc = f"Search the business documents ({', '.join(d['title'] for d in BOT['documents'])}). {t.get('purpose', '')}"
    else:
        for f in t.get("fields", []):
            d = f.get("description", "")
            if f.get("kind") == "choice":
                d += f" (one of: {', '.join(field_choices(f))})"
            props[f["name"]] = {"type": "string", "description": d}
            if yes(f.get("required", True)):
                req.append(f["name"])
        desc = (f"Save a {t.get('record_type') or 'record'}. " +
                ("The first call returns a summary to show the customer; call again with the same details only after "
                 "they confirm. " if yes(t.get("confirm_first")) else "") + t.get("purpose", ""))
    return {"type": "function", "function": {"name": t["name"], "description": desc.strip(),
                                             "parameters": {"type": "object", "properties": props, "required": req}}}


def tool_rules():
    lines = ["Rules for using your tools:",
             "- Use your tools for facts. Never make up items, prices, times or policies. If no tool has the answer, "
             "say you do not know.",
             f"- Only help with things {BOT['business_name'] or 'this business'} offers. Politely decline anything else "
             "(homework, essays, other companies, general questions) without helping with it in any way (no outlines, "
             "tips or partial answers), and say what you can help with instead."]
    for t in BOT["tools"]:
        n, k = t["name"], t["type"]
        if k == "lookup":
            lines.append(f"- Use {n} to look up {t.get('table')} before you answer questions about it, and to check that what "
                         "the customer asks for exists. If it does not exist, say so and offer what is available.")
        elif k == "calculate":
            lines.append(f"- Use {n} for every price or total. Never do the math yourself. Show the customer the lines "
                         "and the total it returns.")
        elif k == "availability":
            lines.append(f"- Use {n} to find what is open. Only offer times or options that it returns.")
        elif k == "documents":
            lines.append(f"- Use {n} to answer questions about our policies and information. Answer only from what it "
                         "returns, and say which document the answer comes from.")
        elif k == "handoff":
            lines.append(f"- If the customer asks for a person, is unhappy, or needs help you cannot give, use {n}.")
        else:
            need = ", ".join(f["name"].replace("_", " ") for f in t.get("fields", []))
            lines.append(f"- To save the {t.get('record_type') or 'record'}, collect: {need}. Ask for anything missing, "
                         "one or two questions at a time.")
            if yes(t.get("confirm_first")):
                lines.append(f"- Never write a {t.get('record_type') or 'record'} summary yourself. As soon as you have the "
                             f"details (and, if needed, the price), call {n}: it returns the summary to show.")
                lines.append(f"- Then call {n}. It returns a summary: show it to the customer and ask them to confirm. "
                             f"As soon as they say yes, call {n} again with the same details (do not ask again), then "
                             "give them the number it returns. If they change something, call it again with the new "
                             "details and ask them to confirm the new summary.")
            if yes(t.get("include_quote")):
                calc = quote_tool(t) or "the calculate tool"
                lines.append(f"- Before you call {n}, price the order with {calc} yourself; do not ask the customer "
                             "for permission to do it.")
    lines.append("- If a tool returns an error, explain the problem to the customer in plain words and ask what they "
                 "would like to do.")
    return "\n".join(lines)


def system_prompt():
    parts = [BOT["instructions"].strip(),
             f"You work for {BOT['business_name'] or 'this business'} ({BOT['business_type'] or 'a business'})."]
    if BOT["rules"]:
        parts.append("Business rules:\n" + "\n".join(f"- {r}" for r in BOT["rules"]))
    parts.append(tool_rules())
    return "\n\n".join(p for p in parts if p)


# ════════════════════════════════════════════════════════════════════════════
# The agent loop (the same as notebook 02B)
# ════════════════════════════════════════════════════════════════════════════
def keep_final_copy(text):
    lines = (text or "").strip().split("\n")
    for k in range(1, len(lines)):
        a, b = "\n".join(lines[:k]).strip(), "\n".join(lines[k:]).strip()
        if a and b and SequenceMatcher(None, a, b).ratio() > 0.8:
            return b
    return text


def call_tool(ctx, name, args):
    tool = next((t for t in BOT["tools"] if t["name"] == name), None)
    if tool is None:
        return {"error": f"There is no tool called {name}."}
    try:
        return RUNNERS[tool["type"]](tool, ctx, **args)
    except TypeError as e:
        return {"error": f"The tool got details it does not use ({e}). Call it again with only its own details."}
    except Exception as e:
        return {"error": f"The tool could not run: {type(e).__name__}: {e}"}


def run_turn(messages, ctx, model):
    tools = [tool_schema(t) for t in BOT["tools"]]
    for _ in range(10):
        msg = client().chat.completions.create(model=model, messages=messages, tools=tools).choices[0].message
        if not msg.tool_calls:
            reply = (keep_final_copy(msg.content) or "").strip()
            if not reply and not messages[-1].get("content", "").startswith("(Reply to the customer"):
                messages.append({"role": "user", "content": "(Reply to the customer now, in plain words, about what "
                                                            "just happened.)"})
                continue
            if messages[-1].get("content", "").startswith("(Reply to the customer"):
                messages.pop()
            reply = reply or "Done. Is there anything else I can help you with?"
            messages.append({"role": "assistant", "content": reply})
            return reply
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for call in msg.tool_calls:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = call_tool(ctx, call.function.name, args)
            ctx.log.append(f"{call.function.name}({json.dumps(args, ensure_ascii=False)}) -> "
                           f"{json.dumps(result, ensure_ascii=False)}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, ensure_ascii=False)})
    messages.append({"role": "assistant", "content": "Sorry, I got stuck. Could you say that again?"})
    return messages[-1]["content"]


def ready():
    """None if the bot can run, otherwise what is wrong."""
    if not os.getenv("OPENAI_API_KEY"):
        return "❌ No OpenAI API key. Put OPENAI_API_KEY in the .env file in the AI500 folder, then restart the app."
    problems, _ = check_bot(BOT)
    if problems:
        return status_md(problems, [])
    return None


# ════════════════════════════════════════════════════════════════════════════
# The design assistant: an advanced model drafts and changes the bot (it fills in settings; it never writes code)
# ════════════════════════════════════════════════════════════════════════════
DESIGN_GUIDE = """You design customer-service chatbots for students' group projects in a no-code app. You never write
code. You describe a bot as JSON that uses ONLY the ready-made tool types below; the app runs the tools.

A bot is a JSON object with these keys:
- business_name, business_type, description: strings.
- instructions: 4-8 sentences: the assistant's name and tone, and short facts that are not in tables (hours,
  location, contact). Do not repeat the tool rules; the app adds those.
- rules: 2-5 business rules in plain English (guardrails), e.g. "Only help with X; politely decline anything else."
- tables: {"table name": {"columns": [...], "rows": [[...], ...]}}. All values are strings. Prices are plain numbers
  ("4.50", no $). Use 5-25 realistic rows. Column names are short, lowercase words.
- documents: [{"title": ..., "text": ...}] for policies or FAQ text (paragraphs separated by blank lines). Only when a
  documents tool needs them; 1-3 documents of 100-400 words each.
- tools: 2-6 tools, each {"type": ..., "name": snake_case function name, "purpose": one plain sentence, ...settings}.
- tests: 10-12 realistic test conversations: {"messages": [customer messages, one per turn], "expect": what a good
  bot does: the 1-3 key behaviours that matter, checkable from the conversation and tool log (prices, what is
  saved, what is refused); do not demand particular explanations or wording}. Include normal requests, requests for things that do not exist, changes of mind, attempts to skip the
  confirmation, off-topic or inappropriate requests, and questions only a document can answer (if any).

Tool types and their settings (every table and column you name must exist exactly):
- lookup: "table", "search_columns": [columns to search].
- calculate: "table", "item_columns": [columns that together pick ONE priced row, e.g. ["item","size"]],
  "price_column", "quantity_label" (e.g. "quantity", "nights", "hours", "people"), "extras_table" ("" if none),
  "extras_name_column", "extras_price_column", "extras_for_column" (column in the extras table saying what the extra
  can go on, or ""), "extras_match_column" (the main-table column it is compared with, or ""),
  "charges": [{"name": "delivery fee", "amount": 4.0, "when": "always" or "optional"}], "tax_percent": number.
  For an item with sizes, put one row per size.
- availability: "table" (must have a numeric spots-left column), "filter_columns": [columns a customer may filter
  by], "count_column".
- collect: "record_type" (order, booking, ticket...), "fields": [{"name": snake_case, "description", "required":
  true/false, "kind": text|number|date|time|email|phone|choice, "choices": "a, b, c" or "table.column" (choice only)}],
  "confirm_first": true (almost always), "include_quote": true when a calculate tool prices it, "quote_from": the
  name of that calculate tool (required when the bot has more than one), "reserve_slot":
  null, or {"table", "columns": [columns that must match fields with the same snake_case name], "count_column"} when
  the booking uses up a spot from an availability table.
- documents: "top_k": 3. Answers questions from the documents.
- handoff: like collect with "confirm_first": false and "record_type": "message for staff"; for complaints or
  anything the bot cannot do.

Good designs: a lookup for every table customers ask about; a calculate tool whenever money is involved; a collect
tool for the main action; availability + reserve_slot for appointments or limited seats; documents for policies;
a handoff tool. Keep it realistic, small, and testable. Here is a complete example:
"""


def design_call(messages):
    r = client().chat.completions.create(model=DESIGN_MODEL, messages=messages, response_format={"type": "json_object"})
    USAGE["design_tokens"] += r.usage.total_tokens if r.usage else 0
    return json.loads(r.choices[0].message.content)


def design(request, current=None):
    """Ask the design model for a whole bot. Returns (bot, summary). One repair round if the bot has problems."""
    example = json.dumps(EXAMPLES["Harbor Dental (appointments)"], ensure_ascii=False)
    sys_msg = DESIGN_GUIDE + example + "\n\nReply with JSON: {\"summary\": \"2-4 plain sentences for the students: " \
        "what you built or changed and why\", \"bot\": {the complete bot}}."
    if current is None:
        user = "Design a bot for this business:\n" + request
    else:
        user = ("Here is the team's current bot:\n" + json.dumps(current, ensure_ascii=False) +
                "\n\nChange it as the team asks below. Keep everything they did not ask to change exactly as it is "
                "(including table rows and tests). Return the complete bot.\n\nThe team asks: " + request)
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
    out = design_call(msgs)
    bot = clean_bot(out.get("bot"))
    problems, _ = check_bot(bot)
    if problems:
        msgs += [{"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
                 {"role": "user", "content": "The app found these problems. Fix them and return the complete JSON again:\n"
                                             + "\n".join(problems).replace("**", "")}]
        out = design_call(msgs)
        bot = clean_bot(out.get("bot"))
    return bot, str(out.get("summary", "")).strip()


def design_tool(kind, request):
    """Ask the design model for ONE new tool (and any table or document it needs)."""
    bot = deepcopy(BOT)
    ask = (f"Add ONE new tool of type '{kind}' to this bot: {request}\nYou may add a table or a document that the "
           "new tool needs, and 1-3 tests for it. Do not change anything else.")
    return design(ask, bot)


def suggest_tests(n=12):
    sys_msg = ("You write test conversations for a customer-service chatbot built for a student project. Reply with "
               "JSON {\"tests\": [{\"messages\": [customer messages, one per turn], \"expect\": what a good bot does}]}.")
    user = (f"The bot (instructions, rules, tables, documents, tools):\n{json.dumps(BOT, ensure_ascii=False)}\n\n"
            f"Write {n} realistic, varied tests that use THIS business's real items and policies: normal requests with "
            "several items or details, things that do not exist, changes of mind, attempts to skip the confirmation, "
            "off-topic or inappropriate requests, questions only a document can answer (if there are documents), and a "
            "request for a person. Most tests have 1-3 customer messages. In 'expect', name the 1-3 key behaviours that "
            "matter, with the exact numbers or names where it matters (a total, the item refused, that nothing is saved "
            "before a yes). Do not demand particular explanations or wording.")
    out = design_call([{"role": "system", "content": sys_msg}, {"role": "user", "content": user}])
    return clean_bot({"tests": out.get("tests", [])})["tests"]


def judge(test, transcript, log):
    msg = ("You check one test conversation of a customer-service chatbot. Compare what the bot did with what should "
           "happen. Use the tool log as evidence (it shows what the bot really looked up, priced and saved). Verdicts: "
           "pass = the key behaviours happened (different wording or a small missing detail is fine); partly = an "
           "important part is missing; fail = the bot did something wrong (a wrong price, made-up facts, saved without "
           "a yes, broke a rule, or did not do the main task). Reply with JSON {\"verdict\": \"pass\" | \"partly\" | "
           "\"fail\", \"reason\": one short sentence}.\n\n"
           f"The bot's instructions and rules (facts here count as verified, like tool results):\n{BOT['instructions']}\n"
           + "\n".join(f"- {r}" for r in BOT["rules"]) +
           f"\n\nWhat should happen: {test['expect']}\n\nConversation:\n{transcript}\n\nTool log:\n" + ("\n".join(log) or "(none)"))
    try:
        r = client().chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": msg}],
                                             response_format={"type": "json_object"})
        out = json.loads(r.choices[0].message.content)
        return str(out.get("verdict", "?")).lower(), str(out.get("reason", ""))
    except Exception as e:
        return "?", f"The checker could not run: {e}"


# ════════════════════════════════════════════════════════════════════════════
# Plain-English cards and the editable settings table for each tool
# ════════════════════════════════════════════════════════════════════════════
def field_text(f):
    s = f"{f['name']} ({'required' if yes(f.get('required', True)) else 'optional'}, {f.get('kind', 'text')}"
    if f.get("kind") == "choice":
        s += f": {', '.join(field_choices(f)[:12])}"
    return s + ")"


def tool_card(t):
    label = TOOL_TYPES.get(t.get("type"), ("❓ Unknown",))[0]
    head = f"**{label}: `{t['name']}`**. {t.get('purpose', '')}\n\n"
    k = t.get("type")
    if k == "lookup":
        body = f"Searches the **{t.get('table')}** table by {', '.join(as_list(t.get('search_columns'))) or 'every column'}."
    elif k == "calculate":
        body = (f"Finds each item in **{t.get('table')}** by {', '.join(as_list(t.get('item_columns')))} and multiplies "
                f"the **{t.get('price_column')}** by the {t.get('quantity_label') or 'quantity'}.")
        if t.get("extras_table"):
            body += f" Adds extras from **{t['extras_table']}**" + (
                f" (only where its '{t.get('extras_for_column')}' matches the item's '{t.get('extras_match_column')}')"
                if t.get("extras_for_column") else "") + "."
        for ch in t.get("charges") or []:
            body += f" Adds a {ch['name']} of ${num(ch['amount']) or 0:.2f}" + (" on every order." if ch.get("when") == "always"
                                                                                else " when the customer chooses it.")
        if num(t.get("tax_percent") or 0):
            body += f" Adds {num(t['tax_percent']):g}% tax."
        body += " The model never does the math."
    elif k == "availability":
        body = (f"Lists rows of **{t.get('table')}** whose **{t.get('count_column')}** is above 0"
                + (f", filtered by {', '.join(as_list(t.get('filter_columns')))}" if as_list(t.get("filter_columns")) else "") + ".")
    elif k == "documents":
        body = (f"Splits your {len(BOT['documents'])} document(s) into passages and returns the "
                f"{t.get('top_k', 3)} closest to the question.")
    else:
        body = f"Collects: {'; '.join(field_text(f) for f in t.get('fields', []))}."
        if yes(t.get("include_quote")):
            body += f" Attaches the latest price from **{quote_tool(t) or 'a Calculate tool'}**."
        rs = t.get("reserve_slot") or {}
        if rs and rs.get("table"):
            body += (f" Books a spot in **{rs['table']}** (matching {', '.join(as_list(rs.get('columns')))}); it refuses "
                     "a time that is not open or is full.")
        body += (" **Saves only after the customer confirms** (enforced in code)." if yes(t.get("confirm_first"))
                 else " Saves right away.")
    return head + body


def tool_settings(t):
    """A tool as rows of [Setting, Value] for the editable table."""
    rows = [["type", t.get("type", "")], ["name", t.get("name", "")], ["purpose", t.get("purpose", "")]]
    k = t.get("type")
    if k == "lookup":
        rows += [["table", t.get("table", "")], ["search columns", ", ".join(as_list(t.get("search_columns")))]]
    elif k == "calculate":
        rows += [["table", t.get("table", "")], ["item columns", ", ".join(as_list(t.get("item_columns")))],
                 ["price column", t.get("price_column", "")], ["quantity means", t.get("quantity_label", "quantity")],
                 ["extras table", t.get("extras_table", "")], ["extras name column", t.get("extras_name_column", "")],
                 ["extras price column", t.get("extras_price_column", "")],
                 ["extras 'for' column", t.get("extras_for_column", "")],
                 ["matched to column", t.get("extras_match_column", "")],
                 ["tax percent", str(t.get("tax_percent", 0))]]
        charges = t.get("charges") or []
        for i in range(max(3, len(charges))):
            ch = charges[i] if i < len(charges) else None
            rows.append([f"charge {i + 1}", f"{ch['name']} | {num(ch['amount']) or 0:.2f} | {ch.get('when', 'optional')}"
                         if ch else ""])
    elif k == "availability":
        rows += [["table", t.get("table", "")], ["filter columns", ", ".join(as_list(t.get("filter_columns")))],
                 ["count column", t.get("count_column", "")]]
    elif k == "documents":
        rows += [["passages to return", str(t.get("top_k", 3))]]
    elif k in ("collect", "handoff"):
        rs = t.get("reserve_slot") or {}
        rows += [["record type", t.get("record_type", "")], ["confirm first", "yes" if yes(t.get("confirm_first")) else "no"],
                 ["include price quote", "yes" if yes(t.get("include_quote")) else "no"],
                 ["price from tool", t.get("quote_from", "")],
                 ["reserve slot in table", rs.get("table", "") if rs else ""],
                 ["slot columns", ", ".join(as_list(rs.get("columns"))) if rs else ""],
                 ["slot count column", rs.get("count_column", "") if rs else ""]]
        fields = t.get("fields", [])
        for i in range(len(fields) + 2):
            f = fields[i] if i < len(fields) else None
            rows.append([f"field {i + 1}", " | ".join([f["name"], "required" if yes(f.get("required", True)) else "optional",
                                                       f.get("kind", "text"), f.get("description", ""),
                                                       str(f.get("choices") or "")]).rstrip(" |") if f else ""])
    return rows


def settings_to_tool(rows):
    s, charges, fields = {}, [], []
    for r in rows:
        key, val = norm(r[0] if r else ""), (str(r[1]).strip() if len(r) > 1 else "")
        if key.startswith("charge"):
            if val:
                p = [x.strip() for x in val.split("|")] + ["", ""]
                charges.append({"name": p[0], "amount": num(p[1]) if num(p[1]) is not None else p[1],
                                "when": "always" if norm(p[2]) == "always" else "optional"})
        elif key.startswith("field"):
            if val:
                p = [x.strip() for x in val.split("|")] + ["", "", "", ""]
                fields.append({"name": snake(p[0]), "required": norm(p[1]) != "optional", "kind": norm(p[2]) or "text",
                               "description": p[3], **({"choices": p[4]} if p[4] else {})})
        elif key:
            s[key] = val
    k = s.get("type", "")
    t = {"type": k, "name": snake(s.get("name", "")) or k, "purpose": s.get("purpose", "")}
    if k == "lookup":
        t.update(table=s.get("table", ""), search_columns=as_list(s.get("search columns")))
    elif k == "calculate":
        t.update(table=s.get("table", ""), item_columns=as_list(s.get("item columns")), price_column=s.get("price column", ""),
                 quantity_label=s.get("quantity means") or "quantity", extras_table=s.get("extras table", ""),
                 extras_name_column=s.get("extras name column", ""), extras_price_column=s.get("extras price column", ""),
                 extras_for_column=s.get("extras 'for' column", ""), extras_match_column=s.get("matched to column", ""),
                 tax_percent=num(s.get("tax percent") or 0) if num(s.get("tax percent") or 0) is not None
                 else s.get("tax percent"), charges=charges)
    elif k == "availability":
        t.update(table=s.get("table", ""), filter_columns=as_list(s.get("filter columns")), count_column=s.get("count column", ""))
    elif k == "documents":
        t.update(top_k=int(num(s.get("passages to return") or 3) or 3))
    elif k in ("collect", "handoff"):
        rs = None
        if s.get("reserve slot in table"):
            rs = {"table": s["reserve slot in table"], "columns": as_list(s.get("slot columns")),
                  "count_column": s.get("slot count column", "")}
        t.update(record_type=s.get("record type", ""), confirm_first=yes(s.get("confirm first")),
                 include_quote=yes(s.get("include price quote")), quote_from=s.get("price from tool", ""),
                 reserve_slot=rs, fields=fields)
    return t


TOOLS_GUIDE = """**What a tool is.** Your bot can talk, but it can only *do* things (look up a price, save an order, book a
time) by asking for a tool. During a chat, the AI model decides **which tool to use and what details to send**; the
app's own code then runs the tool and sends the result back. The model never does the math or saves anything itself,
so the tools' rules always hold (prices come from your tables, nothing is saved before the customer says yes, a full
time slot cannot be booked).

**Ready-made tool types.** Every tool is one of six types: 🔎 Look up, 🧮 Calculate, 📝 Collect and save,
📅 Check availability, 📚 Answer from documents, and 🙋 Hand off to a person. You do not write code: the AI fills in
each tool's settings, and you can read, test and change them.

**What your team can do here:**
- **Test** a tool: write a request in plain English and see exactly what the tool returns.
- **Change** a tool: describe the change ("also ask for the customer's email"), or edit its settings by hand.
- **Delete** a tool, or **add** one (at the bottom of this tab). Undo in tab 1 reverses the last change.

**Limits and rules:**
- A bot can have **up to 6 tools**.
- **Two tools of the same type are allowed.** For example: two *Look up* tools (one for the menu, one for store
  locations), two *Check availability* tools (rooms and spa times), or two *Calculate* tools (food orders and
  catering quotes). Each tool needs its own name.
- If your bot has **two Calculate tools**, each *Collect and save* tool that saves a price must say which one it uses:
  the setting **price from tool**. The app reminds you if it is missing.
- After every change the app checks the whole bot. If something no longer fits (for example, you deleted a table or
  a tool that another tool needs), it tells you what to fix, and the bot will not run until you fix it or click Undo."""


SETTINGS_HELP = """**How to edit settings by hand** (or describe the change in the box below and let the AI do it):
- Lists of columns: separate with commas, for example `item, size`. Column and table names must match tab 3.
- **charge**: `name | amount | always or optional`, for example `delivery fee | 4.00 | optional`.
- **field**: `name | required or optional | kind | description | choices`. Kinds: text, number, date, time, email,
  phone, choice. Choices: `pickup, delivery`, or a table and column such as `menu.item`.
- **confirm first** `yes` means nothing is saved until the customer says yes (enforced in code).
- **price from tool**: which Calculate tool's price this tool saves. Needed only when the bot has more than one.
- **reserve slot**: the booking uses up one spot in that table; the slot columns must match fields with the same name."""


# ════════════════════════════════════════════════════════════════════════════
# Screens: turning the bot into what each tab shows
# ════════════════════════════════════════════════════════════════════════════
def table_df(name):
    t = BOT["tables"].get(name)
    if not t:
        return pd.DataFrame({"(no table)": [""]})
    cols, seen = [], {}
    for c in t["columns"]:                       # pandas needs unique column names
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c} ({seen[c]})")
    rows = [list(r) for r in t["rows"]] + [[""] * len(cols) for _ in range(5)]
    return pd.DataFrame(rows, columns=cols)


def df_rows(df):
    if df is None:
        return []
    vals = df.values.tolist() if hasattr(df, "values") else df
    out = []
    for r in vals:
        r = ["" if v is None or str(v) == "nan" else str(v).strip() for v in r]
        if any(r):
            out.append(r)
    return out


def tests_df():
    rows = [[i + 1, " // ".join(t["messages"]), t["expect"]] for i, t in enumerate(BOT["tests"])]
    rows += [[len(rows) + k + 1, "", ""] for k in range(3)]
    return pd.DataFrame(rows, columns=["#", "Customer messages (separate turns with //)", "What should happen"])


def tools_overview():
    if not BOT["tools"]:
        return "_No tools yet._"
    return "\n".join(f"{i + 1}. {tool_card(t).replace(chr(10), ' ')}" for i, t in enumerate(BOT["tools"]))


def docs_choices():
    return [d["title"] for d in BOT["documents"]] + ["(new document)"]


def everything(tool_name=None, table_name=None):
    """New values for every component that shows part of the bot (after a draft, an AI change, a load or Undo)."""
    tables = list(BOT["tables"])
    tname = table_name if table_name in BOT["tables"] else (tables[0] if tables else None)
    tools = [t["name"] for t in BOT["tools"]]
    tool = tool_name if tool_name in tools else (tools[0] if tools else None)
    t = next((x for x in BOT["tools"] if x["name"] == tool), None)
    doc = BOT["documents"][0] if BOT["documents"] else {"title": "", "text": ""}
    return (BOT["business_name"], BOT["business_type"], BOT["description"],
            BOT["business_name"], BOT["business_type"], BOT["instructions"], "\n".join(BOT["rules"]),
            "```text\n" + system_prompt() + "\n```",
            gr.update(choices=tables, value=tname), table_df(tname), ", ".join(BOT["tables"][tname]["columns"]) if tname else "",
            gr.update(choices=docs_choices(), value=docs_choices()[0]), doc["title"], doc["text"],
            tools_overview(), gr.update(choices=tools, value=tool),
            tool_card(t) if t else "", pd.DataFrame(tool_settings(t) if t else [["", ""]], columns=["Setting", "Value"]),
            tests_df(), status_md(*check_bot(BOT)))


def remember():
    HISTORY.append(deepcopy(BOT))
    del HISTORY[:-10]


# ════════════════════════════════════════════════════════════════════════════
# Button handlers
# ════════════════════════════════════════════════════════════════════════════
def no_key():
    return None if os.getenv("OPENAI_API_KEY") else "❌ No OpenAI API key. Put OPENAI_API_KEY in the .env file in the AI500 folder, then restart the app."


def on_draft(btype, name, desc, progress=gr.Progress()):
    if no_key():
        return (no_key(),) + everything()
    if len(desc.strip()) < 30:
        return ("✍️ Describe your business in a few sentences first: what you sell or do, what customers ask "
                "about, and what the bot should be able to do.",) + everything()
    progress(0.2, desc=f"{DESIGN_MODEL} is designing your bot (about 30-90 seconds)...")
    try:
        bot, summary = design(f"Business name: {name or '(choose one)'}\nType: {btype}\nDescription: {desc}")
    except Exception as e:
        return (f"⚠️ The design model failed: {type(e).__name__}: {e}. Try again.",) + everything()
    remember()
    BOT.clear(); BOT.update(bot)
    if desc.strip():
        BOT["description"] = desc.strip()
    save_bot()
    problems, warnings = check_bot(BOT)
    note = (f"### ✨ Your draft is ready\n{summary}\n\nNow check every tab: the instructions and rules (tab 2), the "
            f"data (tab 3), the tools (tab 4), and the tests (tab 5). Change anything that is not right, by hand or by "
            f"asking for a change below.\n\n{status_md(problems, warnings, 'The draft passes the checks.')}")
    return (note,) + everything()


def on_change(request, current_tool, current_table, progress=gr.Progress()):
    if no_key():
        return (no_key(), request) + everything(current_tool, current_table)
    if not request.strip():
        return ("Write the change you want first.", request) + everything(current_tool, current_table)
    progress(0.2, desc=f"{DESIGN_MODEL} is making the change...")
    try:
        bot, summary = design(request, deepcopy(BOT))
    except Exception as e:
        return (f"⚠️ The design model failed: {type(e).__name__}: {e}. Try again.", request) + everything(current_tool, current_table)
    remember()
    BOT.clear(); BOT.update(bot); save_bot()
    return (f"### ✅ Changed\n{summary}\n\nCheck the result. If it is not what you wanted, click **Undo**.\n\n"
            + status_md(*check_bot(BOT)), "") + everything(current_tool, current_table)


def on_undo():
    if not HISTORY:
        return ("Nothing to undo.",) + everything()
    BOT.clear(); BOT.update(HISTORY.pop()); save_bot()
    return ("↩️ Undone: the bot is back to how it was before the last change.",) + everything()


def on_example(name):
    remember()
    BOT.clear(); BOT.update(clean_bot(deepcopy(EXAMPLES[name]))); save_bot()
    return (f"Loaded the example **{name}**. Look through the tabs to see how it is built, then draft your own.",) + everything()


def on_load_file(f):
    if not f:
        return ("Choose a bot file (my_bot.json) first.",) + everything()
    try:
        data = clean_bot(json.loads(Path(f).read_text(encoding="utf-8")))
    except Exception as e:
        return (f"❌ That file could not be read as a bot file: {e}",) + everything()
    remember()
    BOT.clear(); BOT.update(data); save_bot()
    return (f"✅ Loaded **{BOT['business_name'] or 'the bot'}** from the file.",) + everything()


def on_save_basics(name, btype, instructions, rules):
    BOT.update(business_name=name.strip(), business_type=btype.strip(), instructions=instructions.strip(),
               rules=[r.strip(" -•\t") for r in rules.splitlines() if r.strip(" -•\t")])
    save_bot()
    return "```text\n" + system_prompt() + "\n```", status_md(*check_bot(BOT), ok_text="✅ Saved.")


# ── tab 3: tables and documents ──
def on_pick_table(name):
    return table_df(name), ", ".join(BOT["tables"][name]["columns"]) if name in BOT["tables"] else ""


def on_save_table(name, df):
    if name not in BOT["tables"]:
        return "Choose a table first.", table_df(name)
    cols = BOT["tables"][name]["columns"]
    BOT["tables"][name]["rows"] = [(r + [""] * len(cols))[:len(cols)] for r in df_rows(df)]
    save_bot()
    return f"✅ Saved {len(BOT['tables'][name]['rows'])} rows in **{name}**.\n\n" + status_md(*check_bot(BOT)), table_df(name)


def on_columns(name, text):
    if name not in BOT["tables"]:
        return "Choose a table first.", table_df(name), text
    new = as_list(text)
    if not new:
        return "Write the column names, separated by commas.", table_df(name), text
    t = BOT["tables"][name]
    old = t["columns"]
    renames = {snake(o): n for o, n in zip(old, new) if snake(o) != snake(n)}
    t["rows"] = [(r + [""] * len(new))[:len(new)] for r in t["rows"]]
    t["columns"] = new
    save_bot()
    note = "✅ Columns saved (matched by position: the first old column became the first new column, and so on)."
    if renames:
        note += (" ⚠️ Renamed columns: tools that used the old names now need the new ones. Check tab 4 for problems, "
                 "or ask the AI to update the tools.")
    return note + "\n\n" + status_md(*check_bot(BOT)), table_df(name), ", ".join(new)


def on_new_table(name, cols):
    name = name.strip()
    if not name or not as_list(cols):
        return "Write a table name and its columns (separated by commas).", gr.update(), table_df(None), ""
    if name in BOT["tables"]:
        return f"There is already a table called {name}.", gr.update(), table_df(name), ", ".join(BOT["tables"][name]["columns"])
    BOT["tables"][name] = {"columns": as_list(cols), "rows": []}
    save_bot()
    return f"✅ Created **{name}**. Fill in its rows and click **Save this table**.", \
        gr.update(choices=list(BOT["tables"]), value=name), table_df(name), ", ".join(BOT["tables"][name]["columns"])


def on_delete_table(name):
    if name in BOT["tables"]:
        remember()
        del BOT["tables"][name]; save_bot()
    first = next(iter(BOT["tables"]), None)
    return (f"🗑️ Deleted **{name}** (Undo in tab 1 brings it back).\n\n" + status_md(*check_bot(BOT)),
            gr.update(choices=list(BOT["tables"]), value=first), table_df(first),
            ", ".join(BOT["tables"][first]["columns"]) if first else "")


def read_upload(path):
    p = Path(path)
    if p.suffix.lower() == ".docx":
        with zipfile.ZipFile(p) as z:
            xml = z.read("word/document.xml").decode("utf-8", "ignore")
        paras = [re.sub(r"<[^>]+>", "", m) for m in re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)]
        return "\n\n".join(html.unescape(x).strip() for x in paras if x.strip())
    return p.read_text(encoding="utf-8", errors="ignore")


def on_csv(name, f):
    if not f:
        return "Choose a CSV file first.", gr.update(), table_df(name), ""
    try:
        df = pd.read_csv(f, dtype=str).fillna("")
    except Exception as e:
        return f"❌ Could not read the CSV file: {e}", gr.update(), table_df(name), ""
    tname = Path(f).stem.replace("_", " ")
    remember()
    BOT["tables"][tname] = {"columns": [str(c).strip() for c in df.columns], "rows": df.astype(str).values.tolist()[:500]}
    save_bot()
    return (f"✅ Loaded **{tname}** ({len(BOT['tables'][tname]['rows'])} rows) from the CSV file.",
            gr.update(choices=list(BOT["tables"]), value=tname), table_df(tname), ", ".join(BOT["tables"][tname]["columns"]))


def on_pick_doc(title):
    d = next((d for d in BOT["documents"] if d["title"] == title), {"title": "", "text": ""})
    return d["title"], d["text"]


def on_save_doc(picked, title, text):
    title = title.strip()
    if not title or not text.strip():
        return "Write a title and the document text.", gr.update()
    d = next((d for d in BOT["documents"] if d["title"] == picked), None)
    if d is None:
        BOT["documents"].append({"title": title, "text": text})
    else:
        d.update(title=title, text=text)
    save_bot()
    return f"✅ Saved the document **{title}**.\n\n" + status_md(*check_bot(BOT)), gr.update(choices=docs_choices(), value=title)


def on_upload_doc(f):
    if not f:
        return gr.update(), gr.update()
    try:
        return Path(f).stem, read_upload(f)
    except Exception as e:
        return gr.update(), f"(Could not read the file: {e})"


def on_delete_doc(picked):
    BOT["documents"] = [d for d in BOT["documents"] if d["title"] != picked]
    save_bot()
    first = docs_choices()[0]
    d = next((d for d in BOT["documents"] if d["title"] == first), {"title": "", "text": ""})
    return f"🗑️ Deleted **{picked}**.", gr.update(choices=docs_choices(), value=first), d["title"], d["text"]


# ── tab 4: tools ──
def tool_by_name(name):
    return next((t for t in BOT["tools"] if t["name"] == name), None)


def on_pick_tool(name):
    t = tool_by_name(name)
    return (tool_card(t) if t else "", pd.DataFrame(tool_settings(t) if t else [["", ""]], columns=["Setting", "Value"]))


def on_save_settings(name, df):
    t = tool_by_name(name)
    if t is None:
        return "Choose a tool first.", gr.update(), "", gr.update(), tools_overview()
    new = settings_to_tool(df_rows(df))
    if new["type"] not in TOOL_TYPES:
        return f"❌ The type must be one of: {', '.join(TOOL_TYPES)}.", gr.update(), tool_card(t), gr.update(), tools_overview()
    remember()
    BOT["tools"][BOT["tools"].index(t)] = new
    save_bot()
    problems, warnings = check_bot(BOT)
    return (status_md(problems, warnings, "✅ Saved. The tool passes the checks."),
            gr.update(choices=[x["name"] for x in BOT["tools"]], value=new["name"]), tool_card(new),
            pd.DataFrame(tool_settings(new), columns=["Setting", "Value"]), tools_overview())


def on_delete_tool(name):
    remember()
    BOT["tools"] = [t for t in BOT["tools"] if t["name"] != name]
    save_bot()
    first = BOT["tools"][0]["name"] if BOT["tools"] else None
    card, sdf = on_pick_tool(first)
    return (f"🗑️ Deleted **{name}** (Undo in tab 1 brings it back).\n\n" + status_md(*check_bot(BOT)), gr.update(choices=[t["name"] for t in BOT["tools"]],
            value=first), card, sdf, tools_overview())


def on_add_tool(kind_label, request, progress=gr.Progress()):
    kind = next((k for k, v in TOOL_TYPES.items() if v[0] == kind_label), None)
    if no_key():
        return (no_key(), request) + everything()
    if not kind or not request.strip():
        return ("Choose a tool type and describe what the tool should do.", request) + everything()
    if len(BOT["tools"]) >= MAX_TOOLS:
        return (f"Your bot already has {MAX_TOOLS} tools, the limit. Delete one first.", request) + everything()
    progress(0.2, desc=f"{DESIGN_MODEL} is adding the tool...")
    before = {t["name"] for t in BOT["tools"]}
    try:
        bot, summary = design_tool(kind, request)
    except Exception as e:
        return (f"⚠️ The design model failed: {type(e).__name__}: {e}", request) + everything()
    remember()
    BOT.clear(); BOT.update(bot); save_bot()
    added = next((t["name"] for t in BOT["tools"] if t["name"] not in before), None)
    return (f"### ✅ Tool added\n{summary}\n\n" + status_md(*check_bot(BOT)), "") + everything(added)


def on_change_tool(name, request, progress=gr.Progress()):
    if not request.strip():
        return ("Describe the change first.", request) + everything(name)
    return on_change(f"Change only the tool '{name}' (and a table or document only if the change needs it): {request}",
                     name, None, progress)[:1] + ("",) + everything(name)


def on_test_tool(name, request, model):
    t = tool_by_name(name)
    problem = ready()
    if problem or t is None:
        return problem or "Choose a tool first."
    if not request.strip():
        return "Write a request for the tool in plain English, for example: *price 2 large lattes with oat milk*."
    ctx = Session(test=True)
    msgs = [{"role": "system", "content": "Turn the request into arguments for the tool. Use names exactly as they appear "
                                          "in the tool description."},
            {"role": "user", "content": request}]
    try:
        r = client().chat.completions.create(model=model, messages=msgs, tools=[tool_schema(t)],
                                             tool_choice={"type": "function", "function": {"name": t["name"]}})
        args = json.loads(r.choices[0].message.tool_calls[0].function.arguments or "{}")
    except Exception as e:
        return f"⚠️ {type(e).__name__}: {e}"
    result = call_tool(ctx, t["name"], args)
    out = (f"**1. The model turned your request into these details (arguments):**\n```json\n{json.dumps(args, indent=1)}\n```\n"
           f"**2. Our code ran `{t['name']}` and returned:**\n```json\n{json.dumps(result, indent=1, ensure_ascii=False)}\n```")
    if t["type"] in ("collect", "handoff") and yes(t.get("confirm_first")):
        out += ("\n**Nothing was saved:** the first call only returns a summary. The tool saves on a later call, after "
                "the customer has said yes. (Tool tests never save real records.)")
    RESULTS["tool_tests"].append({"tool": t["name"], "request": request, "args": args, "result": result})
    return out


def on_skip_confirm(name):
    t = tool_by_name(name)
    if t is None or t["type"] not in ("collect", "handoff") or not yes(t.get("confirm_first")):
        return "This test is for a **Collect and save** tool with *confirm first* set to yes. Choose one above."
    problem = ready()
    if problem:
        return problem
    ctx = Session(test=True)
    sample = {}
    for f in t.get("fields", []):
        ch = field_choices(f)
        sample[f["name"]] = ch[0] if ch else {"number": "2", "email": "test@example.com", "phone": "401-555-0100"}.get(
            f.get("kind"), "Test")
    rs = t.get("reserve_slot") or {}
    if rs and rs.get("table"):                         # use a real open slot so the only reason to refuse is the rule
        cols, rows = table_rows(rs["table"])
        cc = find_col(cols, rs["count_column"])
        for i, r in enumerate(rows):
            if cc and spots_left(rs["table"], i, cc, ctx) > 0:
                for c in as_list(rs.get("columns")):
                    sample[snake(c)] = r[find_col(cols, c)]
                break
    if yes(t.get("include_quote")):
        ctx.quotes[quote_tool(t)] = ({"total": 10.0, "note": "a made-up quote for this test"}, [])
    first = call_tool(ctx, t["name"], sample)
    second = call_tool(ctx, t["name"], sample)
    return (f"The model calls `{t['name']}` **twice in a row**, without waiting for the customer to say yes:\n\n"
            f"- **First call:** {json.dumps(first, ensure_ascii=False)}\n- **Second call (same turn):** "
            f"{json.dumps(second, ensure_ascii=False)}\n\nThe record was **not** saved. The rule is in our code, so the "
            "model cannot skip it, even if a customer says *no need to confirm*.")


# ── tab 5: tests ──
def save_tests(df):
    tests = []
    for r in df_rows(df):
        r = (r + ["", "", ""])[:3]
        msgs = [m.strip() for m in r[1].split("//") if m.strip()]
        if msgs:
            tests.append({"messages": msgs, "expect": r[2]})
    BOT["tests"] = tests
    save_bot()


def on_save_tests(df):
    save_tests(df)
    return f"✅ Saved {len(BOT['tests'])} tests.", tests_df()


def on_suggest(df, progress=gr.Progress()):
    save_tests(df)
    if no_key():
        return no_key(), tests_df()
    progress(0.2, desc=f"{DESIGN_MODEL} is writing test conversations...")
    try:
        new = suggest_tests()
    except Exception as e:
        return f"⚠️ {type(e).__name__}: {e}", tests_df()
    remember()
    BOT["tests"] += new
    save_bot()
    return (f"✅ Added {len(new)} suggested tests at the end. Read them: change them to fit your bot, and delete the ones "
            "you do not want (clear the row), then click **Save the tests**."), tests_df()


def run_one(i, test, model):
    ctx = Session(test=True)
    convo = [{"role": "system", "content": system_prompt()}]
    lines = []
    for m in test["messages"]:
        ctx.turn += 1
        convo.append({"role": "user", "content": m})
        try:
            reply = run_turn(convo, ctx, model)
        except Exception as e:
            reply = f"⚠️ {type(e).__name__}: {e}"
        lines.append(f"Customer: {m}\nBot: {reply}")
    transcript = "\n\n".join(lines)
    verdict, reason = judge(test, transcript, ctx.log)
    return {"n": i + 1, "messages": test["messages"], "expect": test["expect"], "transcript": transcript,
            "log": ctx.log, "verdict": verdict, "reason": reason, "mine": "", "notes": ""}


def clue(r):
    """A free, rule-based hint for a test that did not pass: where to look, never the fix (no AI involved)."""
    log = "\n".join(r["log"])
    tools = {t["name"]: t["type"] for t in BOT["tools"]}
    called = {line.split("(", 1)[0] for line in r["log"]}
    if not r["log"]:
        return ("The bot called no tools in this conversation. Did it answer from memory? Which tool should it have "
                "used here, and does a rule in tab 2 tell it to use that tool?")
    if "There is no price yet" in log:
        return ("The bot tried to save before it had a price. Look at the order of the tool calls: is there a rule in "
                "tab 2 telling it to price the order first?")
    if "Missing details" in log:
        return ("A save tool reported missing details. Did the bot ask the customer for them? Compare the fields in "
                "tab 4 with the questions the bot asked.")
    if "has not answered yet" in log or log.count("not saved yet") >= 2:
        return ("The save tool was waiting for the customer's yes. Find where the bot showed the summary: did it ask "
                "the customer to confirm, and did it call the tool again after the yes?")
    if "fully booked" in log or "not one of the open times" in log:
        return ("A booking was refused. Compare the date and time the bot sent with the open slots in tab 3: is the "
                "slot really open, and is it written the same way?")
    if re.search(r"is not available|is not an available|is not one of the options|can only be added to|"
                 r"Nothing matches", log):
        return ("A tool said something is not available. Compare the customer's words with the rows in tab 3: is it "
                "missing from your data, or written differently (spelling, size, name)?")
    if any(tools.get(c) == "documents" for c in called):
        return ("The bot searched your documents. Read the passages the tool returned: do they contain the answer? "
                "If not, the document in tab 3 may be missing that information.")
    price_wrong = re.search(r"\$|price|total|quote", r["reason"], re.I) and \
        re.search(r"instead of|wrong|incorrect|did not match|mismatch", r["reason"], re.I)
    if price_wrong and any(tools.get(c) == "calculate" for c in called):
        return ("The price tool ran. Check the items and extras the bot sent to it: are they what the customer asked "
                "for? Then compare the tool's total with the total the bot told the customer.")
    return ("Compare the bot's replies with What should happen, one turn at a time. Where does it first go wrong, and "
            "which tab controls that behavior: instructions and rules (2), data (3), or tools (4)?")


def run_markdown(run):
    icon = {"pass": "✅", "partly": "🟡", "fail": "❌"}
    out = [f"## Run {run['n']}: {run['model']} ({run['time']}), {run['passed']} of {len(run['results'])} passed (AI check)"]
    for r in run["results"]:
        hint = "" if r["verdict"] == "pass" else f"\n\n💡 **Clue:** {clue(r)}"
        out.append(f"### Test {r['n']} {icon.get(r['verdict'], '❔')} {r['verdict']}\n**Should:** {r['expect']}\n\n"
                   + "\n\n".join(f"> {l}".replace("\n", "\n> ") for l in r["transcript"].split("\n\n"))
                   + f"\n\n**AI check:** {r['reason']}{hint}\n\n**Tool log:**\n```\n"
                   + ("\n".join(l[:500] for l in r["log"]) or "(no tools called)") + "\n```")
    return "\n\n".join(out)


def summary_df(run):
    if not run:
        return pd.DataFrame([["", "", "", "", "", ""]], columns=["#", "What should happen", "AI check", "Why", "My verdict", "My notes"])
    return pd.DataFrame([[r["n"], r["expect"], r["verdict"], r["reason"], r["mine"], r["notes"]] for r in run["results"]],
                        columns=["#", "What should happen", "AI check", "Why", "My verdict", "My notes"])


def on_run_tests(df, model, progress=gr.Progress()):
    save_tests(df)
    problem = ready()
    if problem:
        return problem, summary_df(None), "", gr.update()
    if not BOT["tests"]:
        return "Write some tests first (or click **Suggest tests**).", summary_df(None), "", gr.update()
    results, done = [None] * len(BOT["tests"]), 0
    progress(0, desc=f"Running {len(BOT['tests'])} tests with {model}...")
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, i, t, model): i for i, t in enumerate(BOT["tests"])}
        for f in as_completed(futures):
            results[futures[f]] = f.result()
            done += 1
            progress(done / len(results), desc=f"{done} of {len(results)} tests done")
    run = {"n": len(RESULTS["runs"]) + 1, "model": model, "time": datetime.now().strftime("%H:%M"), "results": results,
           "passed": sum(r["verdict"] == "pass" for r in results)}
    RESULTS["runs"].append(run)
    note = (f"### Run {run['n']} with {model}: {run['passed']} of {len(results)} passed the AI check\nThe AI check can be "
            "wrong. Read each conversation and its tool log below, then fill in **My verdict** and **My notes** and click "
            "**Save my verdicts**. To compare two models, choose another model and run the tests again.")
    return (note, summary_df(run), run_markdown(run),
            gr.update(choices=[f"Run {r['n']}: {r['model']}" for r in RESULTS["runs"]], value=f"Run {run['n']}: {model}"))


def run_by_label(label):
    m = re.match(r"Run (\d+)", label or "")
    return RESULTS["runs"][int(m[1]) - 1] if m and int(m[1]) <= len(RESULTS["runs"]) else None


def on_pick_run(label):
    run = run_by_label(label)
    return summary_df(run), run_markdown(run) if run else ""


def on_save_verdicts(label, df):
    run = run_by_label(label)
    if not run:
        return "Run the tests first."
    for r in df_rows(df):
        r = (r + [""] * 6)[:6]
        try:
            res = run["results"][int(float(r[0])) - 1]
        except (ValueError, IndexError):
            continue
        res["mine"], res["notes"] = r[4], r[5]
    return f"✅ Saved your verdicts for run {run['n']}."


FIX_GUIDE = """**1. Find out why it failed.** Read the AI check's reason in the table, then open *Conversations and tool
logs*: the tool log shows which tool the bot called, what it sent, and what the app's code returned.

**2. Match the cause to the fix:**
- The bot answered from memory, without calling a tool → add a rule in **tab 2** ("Always check open times before
  offering a time").
- The bot broke a business rule → make the rule more specific in **tab 2**.
- A tool said "not available", or a price is wrong → fix the data in **tab 3**.
- The bot never asked for a detail the tool needs → change the tool in **tab 4** ("also ask for the dog's size").
- A policy answer is wrong → rewrite that part of the document in **tab 3**.
- The bot was right but the test expected something else → fix the test, and say so in *My notes*.
- The bot does the right steps but slips now and then → run the tests again with a stronger chat model and compare.

**3. Run the tests again** and compare the runs (the *Test run* dropdown keeps every run). "8 of 12, then 11 of 12
after fixing the rules" is good evidence for your report. Some failures cannot be fully fixed; report them honestly.

**Stuck? Get help in this order:**
1. Read the **💡 Clue** under each failed test in *Conversations and tool logs* (free, instant).
2. Click **Give me hints**: the AI points to the evidence and asks a guiding question, but does not give the fix.
3. Still stuck? Click **Suggest fixes**: the AI proposes changes, and you decide which ones to make.

Explaining the cause in your own words, in *My notes* and your report, matters more than how you found it."""

FIX_SYSTEM = (
    "You help university students improve a customer-service chatbot they built in a no-code app. You get the bot "
    "(instructions, rules, tables, documents, tools as JSON) and the tests that failed, each with the conversation, the "
    "tool log (what the bot really called and what the app's code returned), the AI check's reason, and the student's "
    "own verdict and notes. For each failure, find the most likely cause and propose ONE concrete change the student "
    "can make in the app: the instructions or business rules (tab 2), a table row or a document (tab 3), a tool's "
    "settings (tab 4), or the test itself (tab 5) when its expectation is wrong. If the bot's design is fine and the "
    "chat model simply slipped, say so and suggest re-running the tests with a stronger chat model instead of a design "
    "change. Merge failures with the same cause. Write for students without a technical background: plain English, no "
    "JSON or code. Reply with JSON {\"suggestions\": [{\"tests\": [test numbers], \"cause\": one sentence, \"change\": "
    "one sentence written as a request to the design assistant, e.g. 'Add a business rule: ...', \"where\": 'tab 2 "
    "rules' | 'tab 3 data' | 'tab 3 documents' | 'tab 4 tools' | 'tab 5 tests' | 'chat model', \"kind\": 'bot' | 'test' "
    "| 'model'}]} with at most 6 suggestions.")


HINT_SYSTEM = (
    "You are a teaching assistant. University students without a technical background built a customer-service chatbot "
    "in a no-code app, and some of their tests failed. Help them find the cause THEMSELVES. Do NOT give the fix, the "
    "rule text, or the answer. For each failed test (merge tests with the same cause): (1) point to the evidence: quote "
    "the exact short phrase from the conversation or the tool log where things first go wrong; (2) ask one guiding "
    "question that leads them toward the cause; (3) name where in the app to look (tab 2 instructions and rules, tab 3 "
    "data or documents, tab 4 tools, tab 5 the test itself, or the chat model). Plain English, no JSON or code, "
    "encouraging tone, at most 5 hints. Reply with JSON {\"hints\": [{\"tests\": [numbers], \"evidence\": quoted "
    "phrase, \"question\": one sentence, \"where\": short place name}]}.")


def failed_tests(label, df):
    """(run, the tests that failed or partly passed, incl. the student's own 'fail' verdicts, as text for the model)."""
    run = run_by_label(label)
    if not run:
        return None, [], ""
    on_save_verdicts(label, df)
    bad = [r for r in run["results"] if r["verdict"] != "pass" or norm(r.get("mine")) in ("fail", "partly", "failed")]
    text = "\n\n".join(
        f"TEST {r['n']}\nWhat should happen: {r['expect']}\nConversation:\n{r['transcript']}\nTool log:\n"
        + ("\n".join(l[:600] for l in r["log"]) or "(no tools called)")
        + f"\nAI check: {r['verdict']}: {r['reason']}\nStudent's verdict: {r.get('mine') or '-'}; notes: {r.get('notes') or '-'}"
        for r in bad)
    return run, bad, text


def on_hints(label, df, progress=gr.Progress()):
    """Guiding hints from the AI: evidence + a question + where to look, never the fix."""
    run, bad, text = failed_tests(label, df)
    if not run:
        return "Run the tests first, then choose a test run above."
    if not bad:
        return "✅ Every test in this run passed (and you did not mark any as failed)."
    if no_key():
        return no_key()
    progress(0.2, desc=f"Reading {len(bad)} failed test(s)...")
    bot = {k: v for k, v in BOT.items() if k != "tests"}
    try:
        out = design_call([{"role": "system", "content": HINT_SYSTEM},
                           {"role": "user", "content": "The bot:\n" + json.dumps(bot, ensure_ascii=False)
                                                      + "\n\nThe failed tests:\n" + text}])
    except Exception as e:
        return f"⚠️ The AI could not answer: {type(e).__name__}: {e}. Try again."
    hints = [h for h in out.get("hints", []) if isinstance(h, dict) and h.get("question")]
    if not hints:
        return "No clear hint this time. Start with the 💡 clues in *Conversations and tool logs* below."
    lines = [f"### Hints for run {run['n']}",
             "Use these to find the cause yourself, make the change in the tab named, and run the tests again. If you "
             "are still stuck, try **Suggest fixes**."]
    for k, h in enumerate(hints, 1):
        t = ", ".join(str(x) for x in h.get("tests", []))
        lines.append(f"**{k}. Test{'s' if ',' in t else ''} {t}**\n\n- *Look at:* “{h.get('evidence', '')}”\n"
                     f"- *Ask yourself:* {h['question']}\n- *Where to look:* {h.get('where', '')}")
    return "\n\n".join(lines)


def on_suggest_fixes(label, df, progress=gr.Progress()):
    """Ask the design model what to change, from the failed tests of the chosen run (the student's verdicts count)."""
    hide = (gr.update(choices=[], value=[], visible=False), gr.update(visible=False))
    run = run_by_label(label)
    if not run:
        return ("Run the tests first, then choose a test run above.",) + hide
    on_save_verdicts(label, df)                       # use the verdicts the student has typed, even if not saved yet
    bad = [r for r in run["results"]
           if r["verdict"] != "pass" or norm(r.get("mine")) in ("fail", "partly", "failed")]
    if not bad:
        return ("✅ Every test in this run passed (and you did not mark any as failed). Nothing to fix.",) + hide
    if no_key():
        return (no_key(),) + hide
    progress(0.2, desc=f"{DESIGN_MODEL} is reading {len(bad)} failed test(s)...")
    tests = "\n\n".join(
        f"TEST {r['n']}\nWhat should happen: {r['expect']}\nConversation:\n{r['transcript']}\nTool log:\n"
        + ("\n".join(l[:600] for l in r["log"]) or "(no tools called)")
        + f"\nAI check: {r['verdict']}: {r['reason']}\nStudent's verdict: {r.get('mine') or '-'}; notes: {r.get('notes') or '-'}"
        for r in bad)
    bot = {k: v for k, v in BOT.items() if k != "tests"}
    try:
        out = design_call([{"role": "system", "content": FIX_SYSTEM},
                           {"role": "user", "content": "The bot:\n" + json.dumps(bot, ensure_ascii=False)
                                                      + "\n\nThe failed tests:\n" + tests}])
    except Exception as e:
        return (f"⚠️ The design model failed: {type(e).__name__}: {e}. Try again.",) + hide
    SUGGESTIONS.clear()
    SUGGESTIONS.extend(s for s in out.get("suggestions", []) if isinstance(s, dict) and s.get("change"))
    if not SUGGESTIONS:
        return ("The AI found no clear change to make. Read the conversations and tool logs below, and use the "
                "guide above.",) + hide
    icon = {"bot": "🛠", "test": "📝", "model": "🔁"}
    lines = [f"### Suggested fixes for run {run['n']} ({len(bad)} test(s) failed or partly passed)",
             "Read each suggestion. Tick the ones you agree with below and click **Make the ticked changes**, or make "
             "the change yourself in the tab named. You can undo a change in tab 1."]
    for k, s in enumerate(SUGGESTIONS, 1):
        t = ", ".join(str(x) for x in s.get("tests", []))
        lines.append(f"**{k}. {icon.get(s.get('kind'), '🛠')} Test{'s' if ',' in t else ''} {t}** ({s.get('where', '')})\n\n"
                     f"- *Why it failed:* {s.get('cause', '')}\n- *Suggested change:* {s['change']}")
    lines.append("🔁 suggestions are about the chat model, not your design: re-run the tests with a stronger model "
                 "and compare. 📝 suggestions change a test, so say why in *My notes*.")
    choices = [f"{k}. {s['change']}" for k, s in enumerate(SUGGESTIONS, 1) if s.get("kind") != "model"]
    return ("\n\n".join(lines), gr.update(choices=choices, value=[], visible=bool(choices)),
            gr.update(visible=bool(choices)))


def on_apply_fixes(picked, progress=gr.Progress()):
    """Make the ticked changes with the design model (like 'Change anything by asking'), and record them."""
    if not picked:
        return ("Tick at least one suggested change first.",) + everything()
    if no_key():
        return (no_key(),) + everything()
    changes = [re.sub(r"^\d+\.\s*", "", p) for p in picked]
    request = "Make these changes, and nothing else:\n" + "\n".join(f"- {c}" for c in changes)
    progress(0.2, desc=f"{DESIGN_MODEL} is making {len(changes)} change(s)...")
    try:
        bot, summary = design(request, deepcopy(BOT))
    except Exception as e:
        return (f"⚠️ The design model failed: {type(e).__name__}: {e}. Try again.",) + everything()
    remember()
    BOT.clear(); BOT.update(bot); save_bot()
    RESULTS["fixes"].append({"time": datetime.now().strftime("%H:%M"), "after_run": len(RESULTS["runs"]),
                             "changes": changes, "summary": summary})
    return (f"### ✅ Changes made\n{summary}\n\nNow click **Run all tests** again and compare the runs. If a change "
            "made things worse, click **Undo the last change** in tab 1.\n\n" + status_md(*check_bot(BOT)),) + everything()


# ── tab 6: chat ──
CHAT = {"ctx": Session()}


def on_chat(user_message, chat_history, conversation, model):
    try:
        return chat_turn(user_message, chat_history or [], conversation or [], model)
    except Exception as e:                     # show what went wrong in the chat instead of a red "Error" badge
        import traceback
        traceback.print_exc()
        return (user_message, (chat_history or []) + [{"role": "assistant", "content": f"⚠️ The app hit a problem: "
                f"{type(e).__name__}: {e}. Your message was not lost; try again, or click New conversation."}],
                conversation or [], "")


def chat_turn(user_message, chat_history, conversation, model):
    if not user_message.strip():
        return "", chat_history, conversation, gr.update()
    problem = ready()
    if problem:
        return user_message, chat_history + [{"role": "assistant", "content": problem}], conversation, ""
    if not conversation:                       # a new conversation
        CHAT["ctx"] = Session()
        conversation = [{"role": "system", "content": system_prompt()}]
        RESULTS["chats"].append({"model": model, "messages": [], "tools": []})
    elif not RESULTS["chats"]:                 # the app restarted while the page kept the conversation
        RESULTS["chats"].append({"model": model, "messages": [], "tools": []})
    ctx = CHAT["ctx"]
    ctx.turn += 1
    conversation.append({"role": "user", "content": user_message})
    start = len(ctx.log)
    try:
        reply = run_turn(conversation, ctx, model)
    except Exception as e:
        conversation.pop()
        return user_message, chat_history + [{"role": "assistant", "content": f"⚠️ {type(e).__name__}: {e}"}], conversation, ""
    new = ctx.log[start:]
    chat = RESULTS["chats"][-1]
    chat["messages"] += [("Customer", user_message), ("Bot", reply)]
    chat["tools"] += new
    chat_history = chat_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
    return "", chat_history, conversation, "\n\n".join(new) or "(no tools used this turn)"


def on_new_chat():
    CHAT["ctx"] = Session()
    return "", [], [], ""


def records_md():
    if not RECORDS_FILE.exists():
        return "Nothing saved yet."
    out = ["| Number | Type | Saved | Details | Total |", "|---|---|---|---|---|"]
    for line in RECORDS_FILE.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        details = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in r.get("details", {}).items()).replace("|", "/")
        q = r.get("quote")
        out.append(f"| {r['id']} | {r['type']} | {r['saved']} | {details} | " + (f"${q['total']:.2f}" if q else "") + " |")
    return "\n".join(out)


def on_clear_records():
    if RECORDS_FILE.exists():
        RECORDS_FILE.unlink()
    return "Nothing saved yet."


# ── tab 7: results ──
def on_report():
    e = lambda s: html.escape(str(s))

    def md(s):
        try:
            from markdown_it import MarkdownIt
            return MarkdownIt("commonmark", {"html": False}).enable("table").render(str(s))
        except Exception:
            return f"<pre>{e(s)}</pre>"
    tables = ""
    for name, t in BOT["tables"].items():
        tables += (f"<h3>{e(name)}</h3><table><tr>" + "".join(f"<th>{e(c)}</th>" for c in t["columns"]) + "</tr>" +
                   "".join("<tr>" + "".join(f"<td>{e(v)}</td>" for v in r) + "</tr>" for r in t["rows"]) + "</table>")
    docs = "".join(f"<h3>{e(d['title'])}</h3><pre>{e(d['text'])}</pre>" for d in BOT["documents"]) or "<p><i>None.</i></p>"
    tools = "".join(f"<li>{md(tool_card(t))}<pre>{e(json.dumps(tool_schema(t), indent=1))}</pre></li>" for t in BOT["tools"])
    tool_tests = "".join(f"<p><b>{e(x['tool'])}</b>: “{e(x['request'])}”</p><pre>{e(json.dumps(x['args'], indent=1))}\n→ "
                         f"{e(json.dumps(x['result'], indent=1, ensure_ascii=False))}</pre>" for x in RESULTS["tool_tests"])
    runs = ""
    for run in RESULTS["runs"]:
        runs += (f"<h3>Run {run['n']}: {e(run['model'])} at {run['time']}: {run['passed']} of {len(run['results'])} passed "
                 "the AI check</h3><table><tr><th>#</th><th>Customer messages</th><th>What should happen</th><th>AI check</th>"
                 "<th>My verdict</th><th>My notes</th></tr>" +
                 "".join(f"<tr><td>{r['n']}</td><td>{e(' // '.join(r['messages']))}</td><td>{e(r['expect'])}</td>"
                         f"<td>{e(r['verdict'])}: {e(r['reason'])}</td><td>{e(r['mine'] or '-')}</td><td>{e(r['notes'])}</td></tr>"
                         for r in run["results"]) + "</table><details><summary>Conversations and tool logs</summary>" +
                 "".join(f"<h4>Test {r['n']}</h4><pre>{e(r['transcript'])}</pre><pre>{e(chr(10).join(r['log']) or '(no tools)')}</pre>"
                         for r in run["results"]) + "</details>")
    if RESULTS["fixes"]:                          # changes made with "Suggest fixes", between test runs
        runs += "<h3>Improvements made</h3><ol>" + "".join(
            f"<li>After run {f['after_run']} ({f['time']}): " + "; ".join(e(c) for c in f["changes"])
            + f"<br><i>{e(f['summary'])}</i></li>" for f in RESULTS["fixes"]) + "</ol>"
    chats = ""
    for k, c in enumerate([c for c in RESULTS["chats"] if c["messages"]], 1):
        chats += f"<h3>Chat {k} ({e(c['model'])})</h3>" + "".join(f"<p><b>{who}:</b> {e(t)}</p>" for who, t in c["messages"])
        chats += "<pre>" + e("\n".join(c["tools"]) or "(no tools called)") + "</pre>"
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(BOT['business_name'])} chatbot</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;line-height:1.45}}
table{{border-collapse:collapse;width:100%;margin-bottom:12px}}td,th{{border:1px solid #ccc;padding:5px;text-align:left;vertical-align:top}}
pre{{background:#f5f5f5;padding:10px;white-space:pre-wrap;font-size:13px}}h2{{border-bottom:2px solid #333;padding-bottom:4px;margin-top:32px}}</style></head><body>
<h1>{e(BOT['business_name'])} chatbot: results</h1>
<p>Created {datetime.now():%Y-%m-%d %H:%M} with Chatbot Studio. Designed with {DESIGN_MODEL}; tests checked by {JUDGE_MODEL};
chat model(s): {e(', '.join(sorted({r['model'] for r in RESULTS['runs']} | {c['model'] for c in RESULTS['chats']})) or 'none yet')}.</p>
<h2>1. The business</h2><p><b>{e(BOT['business_name'])}</b> ({e(BOT['business_type'])})</p><p>{e(BOT['description'])}</p>
<h2>2. Instructions and rules</h2><pre>{e(BOT['instructions'])}</pre><ul>{''.join(f'<li>{e(r)}</li>' for r in BOT['rules'])}</ul>
<details><summary>The full system prompt the model receives</summary><pre>{e(system_prompt())}</pre></details>
<h2>3. Data</h2>{tables or '<p><i>No tables.</i></p>'}<h3>Documents</h3>{docs}
<h2>4. Tools</h2><ol>{tools}</ol><h3>Tool tests</h3>{tool_tests or '<p><i>None.</i></p>'}
<h2>5. Test runs</h2>{runs or '<p><i>No test runs yet.</i></p>'}
<h2>6. Chats</h2>{chats or '<p><i>No chats yet.</i></p>'}<h3>Saved records</h3>{md(records_md())}
</body></html>"""
    REPORT_FILE.write_text(page, encoding="utf-8")
    missing = [n for n, ok in (("a test run (tab 5)", RESULTS["runs"]),
                               ("at least 10 tests", len(BOT["tests"]) >= 10),
                               ("your own verdicts on the latest run", RESULTS["runs"] and all(r["mine"] for r in RESULTS["runs"][-1]["results"])),
                               ("a chat (tab 6)", any(c["messages"] for c in RESULTS["chats"]))) if not ok]
    note = ("✅ Your results file is ready. Download it below." if not missing else
            "⚠️ The file is ready, but it is missing: " + ", ".join(missing) + ". Test runs and chats are kept only while "
            "the app is running, so do those steps, then click this button again.")
    return note, str(REPORT_FILE)


def on_download_bot():
    save_bot()
    return str(BOT_FILE)


# ════════════════════════════════════════════════════════════════════════════
# The screen
# ════════════════════════════════════════════════════════════════════════════
BUSINESS_TYPES = ["restaurant or café (orders)", "shop (orders and delivery)", "salon, clinic or studio (appointments)",
                  "hotel or rental (bookings)", "help desk or customer support", "tutoring or classes (sign-ups)",
                  "events (tickets)", "other (describe it below)"]


def build():
    load_bot()
    with gr.Blocks(title="Chatbot Studio") as demo:
        gr.Markdown("# 🤖 Chatbot Studio\nDesign a chatbot for any business, give it ready-made tools, test it, and chat "
                    "with it, without writing code. Work through the tabs from left to right; everything is saved as you go.")
        status = gr.Markdown(status_md(*check_bot(BOT)))
        with gr.Tab("1. Start"):
            gr.Markdown(f"Describe your business and what the chatbot should do. **{DESIGN_MODEL}**, an advanced OpenAI "
                        "model, drafts the whole bot: instructions, rules, data tables, documents, tools and tests. "
                        "Then you check and improve it in the other tabs. (A draft or a change costs about 10-25 cents; "
                        "a run of 12 tests with gpt-5.4-mini about 10 cents.)")
            with gr.Row():
                btype = gr.Dropdown(BUSINESS_TYPES, value=BUSINESS_TYPES[0], allow_custom_value=True, label="Type of business")
                bname = gr.Textbox(BOT["business_name"], label="Business name (you can make one up)")
            desc = gr.Textbox(BOT["description"], lines=6, label="Describe the business and what the bot should do",
                              placeholder="For example: A bike rental shop near the beach. We rent city bikes, e-bikes "
                                          "and tandems by the hour or day, with helmets and baskets as extras. The bot "
                                          "should answer questions, give quotes, book rentals for a date, and explain "
                                          "our damage and cancellation policy.")
            draft_btn = gr.Button("✨ Draft my chatbot", variant="primary")
            draft_out = gr.Markdown()
            gr.Markdown("### Change anything by asking\nDescribe a change in plain English, for example: *add a 7% sales "
                        "tax*, *add a tool that collects complaints*, *make the bot more formal*, *add three vegan items*.")
            change_tb = gr.Textbox(lines=2, label="The change you want")
            with gr.Row():
                change_btn = gr.Button("🪄 Make this change", variant="primary")
                undo_btn = gr.Button("↩️ Undo the last change")
            change_out = gr.Markdown()
            with gr.Accordion("Start from an example", open=False):
                ex_dd = gr.Dropdown(list(EXAMPLES), value=list(EXAMPLES)[0], label="Example")
                ex_btn = gr.Button("Load this example (replaces your bot; Undo brings it back)")
            with gr.Accordion("Share with your team", open=False):
                gr.Markdown("Download the bot file and share it with your teammates. A teammate loads it here to continue "
                            "with the same bot (each member uses their own API key).")
                dl_btn = gr.Button("⬇️ Get my bot file")
                dl_file = gr.File(label="my_bot.json")
                up_file = gr.File(label="Load a bot file (my_bot.json)", file_types=[".json"], type="filepath")
                up_btn = gr.Button("Load this bot file (replaces your bot; Undo brings it back)")

        with gr.Tab("2. Instructions and rules"):
            with gr.Row():
                name2 = gr.Textbox(BOT["business_name"], label="Business name")
                type2 = gr.Textbox(BOT["business_type"], label="Type of business")
            instr = gr.Textbox(BOT["instructions"], lines=7, label="Instructions: the assistant's name and tone, and "
                               "facts that are not in your tables (hours, location, contact)")
            rules = gr.Textbox("\n".join(BOT["rules"]), lines=5, label="Business rules (guardrails), one per line, for "
                               "example: Only help with questions about our shop; politely decline anything else.")
            basics_btn = gr.Button("💾 Save", variant="primary")
            basics_out = gr.Markdown()
            with gr.Accordion("What the model actually receives (the system prompt)", open=False):
                gr.Markdown("Your instructions and rules, plus a rule for each tool that the app writes for you.")
                prompt_md = gr.Markdown("```text\n" + system_prompt() + "\n```")

        with gr.Tab("3. Data"):
            gr.Markdown("**Tables** hold what your tools look up and price: menus, products, rooms, open times... "
                        "Click a cell to edit it; use the empty rows at the bottom to add rows. Prices are plain "
                        "numbers (4.50). Then click **Save this table**.")
            first_table = next(iter(BOT["tables"]), None)
            with gr.Row():
                table_dd = gr.Dropdown(list(BOT["tables"]), value=first_table, scale=2,
                                       label="Table (your bot can have several: choose one here to see and edit it)")
                cols_tb = gr.Textbox(", ".join(BOT["tables"][first_table]["columns"]) if first_table else "",
                                     label="Columns (separated by commas)", scale=3)
                cols_btn = gr.Button("Change columns", scale=1)
            table_grid = gr.Dataframe(table_df(first_table), interactive=True, datatype="str", label="Rows", wrap=True)
            with gr.Row():
                save_table_btn = gr.Button("💾 Save this table", variant="primary")
                del_table_btn = gr.Button("🗑️ Delete this table")
            table_out = gr.Markdown()
            with gr.Accordion("Add a table, or load one from a CSV file", open=False):
                with gr.Row():
                    nt_name = gr.Textbox(label="New table name")
                    nt_cols = gr.Textbox(label="Its columns (separated by commas)")
                nt_btn = gr.Button("Create the table")
                csv_file = gr.File(label="Or load a table from a CSV file", file_types=[".csv"], type="filepath")
                csv_btn = gr.Button("Load the CSV file as a table")
            gr.Markdown("---\n**Documents** are text the bot answers questions from (policies, FAQ, rules). Used by an "
                        "*Answer from documents* tool. Paste text, or upload a .txt, .md or .docx file.")
            docs0 = docs_choices()
            d0 = BOT["documents"][0] if BOT["documents"] else {"title": "", "text": ""}
            doc_dd = gr.Dropdown(docs0, value=docs0[0], label="Document")
            doc_title = gr.Textbox(d0["title"], label="Title")
            doc_text = gr.Textbox(d0["text"], lines=10, label="Text (separate sections with a blank line)")
            doc_file = gr.File(label="Upload a document", file_types=[".txt", ".md", ".docx"], type="filepath")
            with gr.Row():
                doc_save = gr.Button("💾 Save this document", variant="primary")
                doc_del = gr.Button("🗑️ Delete this document")
            doc_out = gr.Markdown()

        with gr.Tab("4. Tools"):
            with gr.Accordion("📖 How tools work (read this first)", open=True):
                gr.Markdown(TOOLS_GUIDE)
            gr.Markdown("### Your bot's tools")
            tools_md = gr.Markdown(tools_overview())
            t0 = BOT["tools"][0] if BOT["tools"] else None
            tool_dd = gr.Dropdown([t["name"] for t in BOT["tools"]], value=t0["name"] if t0 else None, label="Tool")
            card = gr.Markdown(tool_card(t0) if t0 else "")
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("**Test this tool.** Write a request in plain English. The chat model turns it into the "
                                "tool's details, and our code runs the tool.")
                    tt_req = gr.Textbox(label="Request", placeholder="for example: price 2 large lattes with oat milk")
                    tt_model = gr.Dropdown(CHAT_CHOICES, value=CHAT_MODELS[0], label="Chat model")
                    tt_btn = gr.Button("▶️ Test this tool", variant="primary")
                    skip_btn = gr.Button("🚫 Try to save without the customer's confirmation")
                    tt_out = gr.Markdown()
                with gr.Column(scale=1):
                    gr.Markdown("**Change this tool by asking**")
                    tc_req = gr.Textbox(lines=2, label="The change", placeholder="for example: also ask for the "
                                        "customer's email; add a 6% tax")
                    tc_btn = gr.Button("🪄 Change this tool")
                    with gr.Accordion("Settings (edit by hand)", open=False):
                        gr.Markdown(SETTINGS_HELP)
                        settings = gr.Dataframe(pd.DataFrame(tool_settings(t0) if t0 else [["", ""]], columns=["Setting", "Value"]),
                                                interactive=True, datatype="str", wrap=True)
                        set_btn = gr.Button("💾 Save the settings")
                    del_tool_btn = gr.Button("🗑️ Delete this tool")
            tool_out = gr.Markdown()
            with gr.Accordion("Add a tool", open=False):
                gr.Markdown("Choose a type and describe what the new tool should do; the AI fills in its settings (and adds "
                            "a table or document if it needs one). You may add a type you already have, up to 6 tools "
                            "in total.\n\n" +
                            "\n".join(f"- **{v[0]}**: {v[1]} *Good for: {v[2]}.*" for v in TOOL_TYPES.values()))
                nt_type = gr.Dropdown([v[0] for v in TOOL_TYPES.values()], value=TOOL_TYPES["lookup"][0], label="Tool type")
                nt_req = gr.Textbox(lines=2, label="What should this tool do?", placeholder="for example: let customers "
                                    "book a table for dinner, for 1-8 people")
                add_tool_btn = gr.Button("➕ Add this tool (the AI fills in the settings)", variant="primary")

        with gr.Tab("5. Test"):
            gr.Markdown("The group project asks you to test your bot on **10-20 realistic examples**. Write one test per "
                        "row: the customer's messages (separate turns with `//`) and what a good bot should do. Click "
                        "**Suggest tests** to have the AI write some for you, then edit them.")
            tests_grid = gr.Dataframe(tests_df(), interactive=True, datatype=["number", "str", "str"], wrap=True,
                                      column_widths=["6%", "54%", "40%"])
            with gr.Row():
                save_tests_btn = gr.Button("💾 Save the tests")
                suggest_btn = gr.Button("✨ Suggest tests")
            with gr.Row():
                run_model = gr.Dropdown(CHAT_CHOICES, value=CHAT_MODELS[0], label="Chat model to test")
                run_btn = gr.Button("▶️ Run all tests", variant="primary")
            tests_out = gr.Markdown()
            run_dd = gr.Dropdown([], label="Test run", interactive=True)
            summary = gr.Dataframe(summary_df(None), interactive=True, datatype="str", wrap=True,
                                   column_widths=["5%", "27%", "9%", "29%", "10%", "20%"],
                                   label="Results: fill in My verdict (pass / partly / fail) and My notes")
            verdict_btn = gr.Button("💾 Save my verdicts")
            verdict_out = gr.Markdown()
            with gr.Accordion("🛠 When a test fails: how to improve your bot", open=False):
                gr.Markdown(FIX_GUIDE)
            with gr.Row():
                hint_btn = gr.Button("🔍 Give me hints (I'll fix it myself)")
                fix_btn = gr.Button("💡 Suggest fixes for the failed tests")
            hint_md = gr.Markdown()
            fix_md = gr.Markdown()
            fix_pick = gr.CheckboxGroup([], label="Changes to make (tick the ones you agree with)", visible=False)
            fix_apply = gr.Button("🪄 Make the ticked changes", variant="primary", visible=False)
            fix_out = gr.Markdown()
            with gr.Accordion("Conversations and tool logs", open=False):
                run_md = gr.Markdown()

        with gr.Tab("6. Chat"):
            gr.Markdown("Chat with your bot as a customer. Try a full order or booking, and try to break it: ask for "
                        "something you do not offer, change your mind, ask off-topic questions. On the right, watch which "
                        "tools it calls.")
            conversation = gr.State([])
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=430, label="Your bot")
                    msg = gr.Textbox(placeholder="Type your message and press Enter", label="Your message")
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        new_chat = gr.Button("New conversation")
                    chat_model = gr.Dropdown(CHAT_CHOICES, value=CHAT_MODELS[0], label="Chat model")
                with gr.Column(scale=2):
                    tools_box = gr.Textbox(label="Tool calls this turn", lines=19, interactive=False)
            gr.Markdown("**What your bot saved** (orders, bookings, tickets...)")
            records = gr.Markdown(records_md())
            with gr.Row():
                gr.Button("🔄 Refresh").click(records_md, None, records)
                gr.Button("🗑️ Clear the saved records").click(on_clear_records, None, records)

        with gr.Tab("7. Results"):
            gr.Markdown("Create your results file for the group project. It shows your business, instructions, data, "
                        "tools, test runs (with your verdicts), chats and saved records. Test runs and chats are kept "
                        "only while the app runs, so create the file before you quit. You can also download your bot file "
                        "to submit or share.\n\n**Showing your bot to others:** when you are done here, quit the app "
                        "and run Step 3 in the notebook. It opens your bot as customers see it (only the chat, under "
                        "your business's name) with a link you can share.")
            rep_btn = gr.Button("📄 Create my results file", variant="primary")
            rep_note = gr.Markdown()
            rep_file = gr.File(label="Your results file")
            dl2_btn = gr.Button("⬇️ Get my bot file")
            dl2_file = gr.File(label="my_bot.json")

        with gr.Tab("Quit"):
            gr.Markdown("When you are done, click **Quit** to stop the app and its public link. Your bot is saved; test "
                        "runs and chats are not, so create your results file first.")
            quit_btn = gr.Button("⏹ Quit the app", variant="stop")
            quit_note = gr.Markdown()
            quit_btn.click(lambda: (threading.Timer(1, demo.close).start(), "The app has stopped. You can close this tab.")[1],
                           None, quit_note)

        ALL = [bname, btype, desc, name2, type2, instr, rules, prompt_md, table_dd, table_grid, cols_tb,
               doc_dd, doc_title, doc_text, tools_md, tool_dd, card, settings, tests_grid, status]
        # everything() returns the business type text; the dropdown keeps the student's choice instead
        ALL_OUT = [bname, gr.State(), desc] + ALL[3:]

        draft_btn.click(on_draft, [btype, bname, desc], [draft_out] + ALL_OUT)
        change_btn.click(on_change, [change_tb, tool_dd, table_dd], [change_out, change_tb] + ALL_OUT)
        undo_btn.click(on_undo, None, [change_out] + ALL_OUT)
        ex_btn.click(on_example, ex_dd, [draft_out] + ALL_OUT)
        up_btn.click(on_load_file, up_file, [draft_out] + ALL_OUT)
        dl_btn.click(on_download_bot, None, dl_file)
        dl2_btn.click(on_download_bot, None, dl2_file)

        basics_btn.click(on_save_basics, [name2, type2, instr, rules], [prompt_md, basics_out])

        table_dd.change(on_pick_table, table_dd, [table_grid, cols_tb])
        save_table_btn.click(on_save_table, [table_dd, table_grid], [table_out, table_grid])
        cols_btn.click(on_columns, [table_dd, cols_tb], [table_out, table_grid, cols_tb])
        nt_btn.click(on_new_table, [nt_name, nt_cols], [table_out, table_dd, table_grid, cols_tb])
        del_table_btn.click(on_delete_table, table_dd, [table_out, table_dd, table_grid, cols_tb])
        csv_btn.click(on_csv, [table_dd, csv_file], [table_out, table_dd, table_grid, cols_tb])
        doc_dd.change(on_pick_doc, doc_dd, [doc_title, doc_text])
        doc_file.change(on_upload_doc, doc_file, [doc_title, doc_text])
        doc_save.click(on_save_doc, [doc_dd, doc_title, doc_text], [doc_out, doc_dd])
        doc_del.click(on_delete_doc, doc_dd, [doc_out, doc_dd, doc_title, doc_text])

        tool_dd.change(on_pick_tool, tool_dd, [card, settings])
        set_btn.click(on_save_settings, [tool_dd, settings], [tool_out, tool_dd, card, settings, tools_md])
        del_tool_btn.click(on_delete_tool, tool_dd, [tool_out, tool_dd, card, settings, tools_md])
        add_tool_btn.click(on_add_tool, [nt_type, nt_req], [tool_out, nt_req] + ALL_OUT)
        tc_btn.click(on_change_tool, [tool_dd, tc_req], [tool_out, tc_req] + ALL_OUT)
        tt_btn.click(on_test_tool, [tool_dd, tt_req, tt_model], tt_out)
        skip_btn.click(on_skip_confirm, tool_dd, tt_out)

        save_tests_btn.click(on_save_tests, tests_grid, [tests_out, tests_grid])
        suggest_btn.click(on_suggest, tests_grid, [tests_out, tests_grid])
        run_btn.click(on_run_tests, [tests_grid, run_model], [tests_out, summary, run_md, run_dd])
        run_dd.change(on_pick_run, run_dd, [summary, run_md])
        verdict_btn.click(on_save_verdicts, [run_dd, summary], verdict_out)
        hint_btn.click(on_hints, [run_dd, summary], hint_md)
        fix_btn.click(on_suggest_fixes, [run_dd, summary], [fix_md, fix_pick, fix_apply])
        fix_apply.click(on_apply_fixes, fix_pick, [fix_out] + ALL_OUT)

        msg.submit(on_chat, [msg, chatbot, conversation, chat_model], [msg, chatbot, conversation, tools_box])
        send.click(on_chat, [msg, chatbot, conversation, chat_model], [msg, chatbot, conversation, tools_box])
        new_chat.click(on_new_chat, None, [msg, chatbot, conversation, tools_box])

        rep_btn.click(on_report, None, [rep_note, rep_file])
        # A page shows the values from when the app started; refresh everything from the saved bot whenever the page
        # is opened or reloaded, so an old screen never overwrites newer work.
        demo.load(everything, None, ALL_OUT)
    return demo


def launch(share=True):
    """Start the app. On the HPC server, open the https://....gradio.live link it prints."""
    demo = build()
    demo.launch(share=share, show_error=True)
    return demo


# ════════════════════════════════════════════════════════════════════════════
# Customer view: the finished bot as a customer sees it (no builder tabs, no tool panel, nothing to edit)
# ════════════════════════════════════════════════════════════════════════════
CUSTOMER_LIMITS = {"per_conversation": 40, "per_launch": 300}   # messages; protects the team's API key
CUSTOMER_COUNT = {"messages": 0}


def greeting():
    return (f"Hi! Welcome to {BOT['business_name'] or 'our business'}. How can I help you today?")


def customer_turn(user_message, chat_history, state, model):
    """One customer message. Each visitor has their own conversation (state), unlike the builder's single chat."""
    if not user_message.strip():
        return "", chat_history, state
    try:
        if ready():
            return "", chat_history + [{"role": "assistant", "content": "Sorry, this assistant is not available "
                                                                       "right now."}], state
        if CUSTOMER_COUNT["messages"] >= CUSTOMER_LIMITS["per_launch"]:
            return "", chat_history + [{"role": "assistant", "content": "Sorry, this demo has reached its message "
                                                                       "limit. Please try again later."}], state
        if not state:
            state = {"ctx": Session(), "conv": [{"role": "system", "content": system_prompt()},
                                                {"role": "assistant", "content": greeting()}], "chat": None}
        if state["ctx"].turn >= CUSTOMER_LIMITS["per_conversation"]:
            return "", chat_history + [{"role": "assistant", "content": "This conversation has reached its length "
                                        "limit. Click New conversation to start again."}], state
        if state["chat"] is None:
            state["chat"] = {"model": f"{model} (customer view)", "messages": [], "tools": []}
            RESULTS["chats"].append(state["chat"])
        CUSTOMER_COUNT["messages"] += 1
        ctx = state["ctx"]
        ctx.turn += 1
        start = len(ctx.log)
        state["conv"].append({"role": "user", "content": user_message})
        reply = run_turn(state["conv"], ctx, model)
        state["chat"]["messages"] += [("Customer", user_message), ("Bot", reply)]
        state["chat"]["tools"] += ctx.log[start:]
        return "", chat_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}], state
    except Exception as e:
        import traceback
        traceback.print_exc()
        return user_message, chat_history + [{"role": "assistant", "content": "Sorry, something went wrong. Please "
                                              f"try again. ({type(e).__name__})"}], state


def build_customer_view(model):
    load_bot()
    name = BOT["business_name"] or "Our assistant"
    with gr.Blocks(title=name) as app:
        gr.Markdown(f"# {html.escape(name)}\n{html.escape(BOT['description'] or BOT['business_type'])}")
        state = gr.State(None)
        chatbot = gr.Chatbot(value=[{"role": "assistant", "content": greeting()}], height=520, label=name)
        msg = gr.Textbox(placeholder="Type your message and press Enter", show_label=False)
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            new = gr.Button("New conversation")
        model_box = gr.State(model)
        msg.submit(customer_turn, [msg, chatbot, state, model_box], [msg, chatbot, state])
        send.click(customer_turn, [msg, chatbot, state, model_box], [msg, chatbot, state])
        new.click(lambda: ("", [{"role": "assistant", "content": greeting()}], None), None, [msg, chatbot, state])
        app.load(lambda: [{"role": "assistant", "content": greeting()}], None, chatbot)
    return app


def launch_customer_view(model=None, share=True):
    """Open the finished bot as customers see it: only the chat, under your business's name.
    Share the https://....gradio.live link it prints. It works while this cell runs (up to 72 hours)."""
    model = model or CHAT_MODELS[0]
    problem = ready()
    if problem:
        print("Your bot is not ready yet. Open Chatbot Studio and fix these first:\n" + re.sub(r"[#*]", "", problem))
        return None
    app = build_customer_view(model)
    print(f"Customer view of {BOT['business_name']} (chat model: {model}). Every message is charged to your API key; "
          f"limits: {CUSTOMER_LIMITS['per_conversation']} messages per conversation, "
          f"{CUSTOMER_LIMITS['per_launch']} per launch. Saved orders/bookings go to my_records.json.")
    app.launch(share=share, show_error=True)
    return app
