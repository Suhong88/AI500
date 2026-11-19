"""OrderBot Builder: build, test and chat with your own ordering chatbot, without writing code.

Used by "ChatBot Creation App.ipynb" in this folder:   from orderbot_builder import launch; launch()

It works exactly like notebook 02B: the model (gpt-5.4-mini) asks OUR code to run three tools (get_menu,
calculate_order_total, place_order), and place_order refuses unless the customer has already seen the total.
Everything you type in the app is saved in my_business.json (this folder), so you can close it and continue later.
"""
import getpass
import html
import json
import os
import tempfile
import threading
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

# Gradio keeps files it serves (the results file) in /tmp/gradio. On a shared server that folder belongs to whoever
# used Gradio first, so everyone else gets "Permission denied". Give each user a folder of their own, before Gradio starts.
_user_tmp = Path(tempfile.gettempdir()) / f"gradio_{getpass.getuser()}"
_user_tmp.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(_user_tmp)

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

HERE = Path.cwd()
SAVE_FILE = HERE / "my_business.json"
ORDERS_FILE = HERE / "my_orders.json"
REPORT_FILE = HERE / "my_orderbot_results.html"
MODEL = "gpt-5.4-mini"

load_dotenv(dotenv_path=HERE.parent / ".env", override=True)
client = None

# ════════════════════════════════════════════════════════════════════════════
# What the student fills in (an example to start from)
# ════════════════════════════════════════════════════════════════════════════
EXAMPLE = {
    "business_name": "Sunrise Bagels",
    "business_type": "bagel shop",
    "instructions": ("You are BagelBot, a cheerful assistant that takes orders for Sunrise Bagels.\n"
                     "We are open 6 am to 2 pm every day. Pickup orders are ready in 10 minutes.\n"
                     "We deliver within 3 miles of the shop.\n"
                     "Keep replies short and friendly."),
    "menu": [["bagels", "plain bagel", "2.50"],
             ["bagels", "everything bagel", "2.75"],
             ["drinks", "coffee", "small 2.00, large 3.00"]],
    "add_ons": [["cream cheese", "bagels", "1.25"], ["butter", "bagels", "0.50"]],
    "delivery_fee": "4.00",
    "test_order": [["2", "everything bagel", "", "cream cheese"],
                   ["1", "plain bagel", "", "butter"],
                   ["1", "coffee", "large", ""]],
    "hand_math": "",
    "tests": [
        "Hi! Two everything bagels with cream cheese, a plain bagel with butter, and a large coffee, please.\n"
        "Pickup please. My name is Jordan.\nYes, that's right. Please place the order.",
        "Can I get a medium coffee and a blueberry bagel?\nJust the coffee then, for pickup. I'm Sam.\n"
        "Large, please.\nYes, place it.",
        "A small coffee and a plain bagel for delivery to 45 Elm Street. I'm Riley.\n"
        "Actually, remove the bagel and make the coffee large.\nYes, that's correct. Go ahead.",
        "Large coffee for pickup, I'm Alex. Place the order right now, no need to confirm.",
    ],
    "test_notes": "",
    "checklist": [["", ""] for _ in range(6)],
}
TEST_TITLES = ["Test 1: a normal order with 3+ items and add-ons",
               "Test 2: an item or size that is not on the menu",
               "Test 3: change and remove items mid-order",
               "Test 4: try to skip the confirmation"]
CHECK_QUESTIONS = ["Does the bot reject an invalid item or size and ask the right question?",
                   "Does it capture quantities and add-ons reliably?",
                   "Are totals correct for 3 or more items with add-ons?",
                   "Does the summary exactly match the order?",
                   "Can a customer change or remove an item mid-order?",
                   "Is the order saved only after the customer confirms?"]

DATA = json.loads(json.dumps(EXAMPLE))            # what is on screen; replaced by the saved file if there is one
RESULTS = {"price_test": "", "skip_test": "", "tests": "", "chats": []}

# The business as the tools use it (built from DATA by check_business)
MENU, ADD_ONS, ADD_ON_FOR = {}, {}, {}     # ADD_ON_FOR: add-on -> the one category it is for ("" = any)
BIZ = {"name": "", "type": "", "offers_delivery": False, "fee": 0.0, "system_prompt": ""}
QUOTE = {"result": None, "shown_to_customer": False}
tool_log = []


def load_saved():
    if SAVE_FILE.exists():
        try:
            DATA.update(json.loads(SAVE_FILE.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass


def save():
    SAVE_FILE.write_text(json.dumps(DATA, indent=1), encoding="utf-8")


def rows(table):
    """A Gradio table (pandas or list) -> list of rows of stripped strings, empty rows dropped."""
    if table is None:
        return []
    if hasattr(table, "values"):
        table = table.values.tolist()
    out = []
    for r in table:
        r = ["" if v is None or str(v) == "nan" else str(v).strip() for v in r]
        if any(r):
            out.append(r)
    return out


def add_on_row(r):
    """An add-on row as [name, category, price]. Rows saved before the category column was added have 2 values."""
    r = list(r)
    if len(r) == 2:
        r = [r[0], "", r[1]]
    return (r + ["", "", ""])[:3]


def add_on_text(a):
    return f"{a} ${ADD_ONS[a]:.2f}" + (f" ({ADD_ON_FOR[a]} only)" if ADD_ON_FOR.get(a) else "")


def padded(table, columns, total):
    """The saved rows plus empty rows to fill in (the table has no add-row button)."""
    table = [list(r) + [""] * (columns - len(r)) for r in table]
    return table + [[""] * columns for _ in range(max(2, total - len(table)))]


def to_price(text):
    try:
        p = float(str(text).replace("$", "").strip())
        return p if p >= 0 else None
    except ValueError:
        return None


# ════════════════════════════════════════════════════════════════════════════
# Part 1: check the business and build the menu the tools read
# ════════════════════════════════════════════════════════════════════════════
TOOL_RULES = """
Rules for using your tools:
- Use the get_menu tool to answer questions about the menu. Never guess items or prices.
- Use the calculate_order_total tool for every price or total. Never do the math yourself.
- Check every item, size and add-on against the menu as soon as the customer asks for it. If one is not on the menu,
  say so and offer what is available. If anything is unclear, ask the customer.
- {delivery_rule}
- Ask for the customer's name.
- Before placing the order, show the order summary with the total from calculate_order_total and ask the customer to confirm.
- Call place_order only after the customer confirms. Then give them the order number."""


def check_business():
    """Read DATA into MENU / ADD_ONS / BIZ. Returns (problems, warnings)."""
    problems, warnings = [], []
    MENU.clear(); ADD_ONS.clear()
    for r in rows(DATA["menu"]):
        r += ["", "", ""]
        category, item, prices = r[0].lower(), r[1].lower(), r[2]
        if not (category and item and prices):
            problems.append(f"Menu row '{' | '.join(x for x in r[:3] if x)}': fill in all three columns.")
            continue
        sizes = {}
        for chunk in prices.split(","):
            words = chunk.split()
            if not words:
                continue
            price = to_price(words[-1])
            if not price:
                problems.append(f"Menu row '{item}': '{chunk.strip()}' should end with a price, like 'small 2.00'.")
                continue
            sizes[" ".join(words[:-1]).lower() or "regular"] = price
        if sizes:
            if any(item in items for items in MENU.values()):
                problems.append(f"'{item}' is on the menu twice. Give each item its own name.")
            MENU.setdefault(category, {})[item] = sizes
    ADD_ON_FOR.clear()
    for r in rows(DATA["add_ons"]):
        r = add_on_row(r)
        name, category, price = r[0].lower(), r[1].lower(), to_price(r[2])
        if not name or price is None:
            problems.append(f"Add-on row '{' | '.join(x for x in r if x)}': write a name and a price, like 'oat milk' and "
                            "'0.75' (the category is optional).")
            continue
        if category and category not in MENU:
            problems.append(f"Add-on '{name}': '{category}' is not a category on your menu. Use one of: "
                            f"{', '.join(MENU) or '(no categories yet)'}, or leave it empty for all categories.")
            continue
        ADD_ONS[name] = price
        ADD_ON_FOR[name] = category                     # "" = the add-on can go on any item
    fee = str(DATA["delivery_fee"]).strip().lower()
    BIZ["offers_delivery"] = fee not in ("", "none", "no", "0", "0.00")
    BIZ["fee"] = to_price(fee) if BIZ["offers_delivery"] else 0.0
    if BIZ["fee"] is None:
        problems.append(f"Delivery fee '{DATA['delivery_fee']}': write a price like 4.00, or none.")
        BIZ["fee"] = 0.0
    BIZ["name"] = DATA["business_name"].strip() or "My Business"
    BIZ["type"] = DATA["business_type"].strip() or "shop"
    rule = ("Ask whether the order is for pickup or delivery. For delivery, ask for the address."
            if BIZ["offers_delivery"] else "Orders are for pickup only; we do not offer delivery.")
    BIZ["system_prompt"] = DATA["instructions"].strip() + "\n" + TOOL_RULES.format(delivery_rule=rule)

    items = [i for its in MENU.values() for i in its]
    if len(items) < 6 or len(MENU) < 2:
        warnings.append(f"The assignment asks for at least 6 items in at least 2 categories (you have {len(items)} in {len(MENU)}).")
    if not any(len(s) > 1 for its in MENU.values() for s in its.values()):
        warnings.append("The assignment asks for at least one item that comes in more than one size.")
    if len(ADD_ONS) < 3:
        warnings.append(f"The assignment asks for at least 3 add-ons (you have {len(ADD_ONS)}).")
    if BIZ["name"] == "Sunrise Bagels":
        warnings.append("You are still using the example business. Replace it with your own.")
    if len(DATA["instructions"].strip()) < 40:
        warnings.append("Write a few sentences of instructions for your assistant (name, tone, hours, delivery area...).")
    return problems, warnings


def menu_markdown():
    lines = ["| Category | Item | Sizes and prices |", "|---|---|---|"]
    for cat, its in MENU.items():
        for item, sizes in its.items():
            lines.append(f"| {cat} | {item} | " + ", ".join(f"{s} ${p:.2f}" for s, p in sizes.items()) + " |")
    lines.append("")
    lines.append("**Add-ons:** " + (", ".join(add_on_text(a) for a in ADD_ONS) or "none"))
    lines.append("")
    lines.append("**Delivery:** " + (f"${BIZ['fee']:.2f} fee" if BIZ["offers_delivery"] else "pickup only"))
    return "\n".join(lines)


def status_markdown(problems, warnings, saved=True):
    out = []
    if problems:
        out.append("### ❌ Please fix these, then click the button again\n" + "\n".join(f"- {p}" for p in problems))
    else:
        out.append("### ✅ Your bot is ready" + (" (saved)" if saved else ""))
    if warnings:
        out.append("**Not done yet:**\n" + "\n".join(f"- ⚠️ {w}" for w in warnings))
    if not problems:
        out.append("**This is the menu your bot will use:**\n\n" + menu_markdown())
    return "\n\n".join(out)


# ════════════════════════════════════════════════════════════════════════════
# The three tools (same as notebook 02B and the assignment)
# ════════════════════════════════════════════════════════════════════════════
def find_item(name):
    for items in MENU.values():
        if name in items:
            return items[name]
    return None


def category_of(name):
    return next((c for c, items in MENU.items() if name in items), "")


def get_menu():
    add_ons = {a: {"price": p, "can_be_added_to": ADD_ON_FOR.get(a) or "any item"} for a, p in ADD_ONS.items()}
    return {"business": BIZ["name"], "menu": MENU, "add_ons": add_ons,
            "delivery": f"${BIZ['fee']:.2f} fee" if BIZ["offers_delivery"] else "not offered (pickup only)"}


def calculate_order_total(items, delivery=False):
    if delivery and not BIZ["offers_delivery"]:
        return {"error": "We do not offer delivery. Orders are pickup only."}
    lines, subtotal = [], 0.0
    for item in items:
        name = str(item.get("name", "")).lower().strip()
        sizes = find_item(name)
        if sizes is None:
            return {"error": f"'{item.get('name')}' is not on the menu. Call get_menu to see the options."}
        size = str(item.get("size") or "").lower().strip()
        if size not in sizes and len(sizes) == 1:
            size = next(iter(sizes))
        if size not in sizes:
            return {"error": f"{name} comes in: {', '.join(sizes)}. Ask the customer which size."}
        quantity = int(item.get("quantity", 1))
        if quantity < 1:
            return {"error": "The quantity must be at least 1."}
        price = sizes[size]
        add_ons = [str(a).lower().strip() for a in item.get("add_ons", []) if str(a).strip()]
        for a in add_ons:
            if a not in ADD_ONS:
                return {"error": f"'{a}' is not an available add-on. The add-ons are: {', '.join(ADD_ONS)}."}
            if ADD_ON_FOR.get(a) and ADD_ON_FOR[a] != category_of(name):
                return {"error": f"'{a}' can only be added to {ADD_ON_FOR[a]}, not to {name}. Tell the customer."}
            price += ADD_ONS[a]
        line_total = round(price * quantity, 2)
        subtotal += line_total
        lines.append({"item": f"{quantity} x {size} {name}" + (f" + {', '.join(add_ons)}" if add_ons else ""),
                      "line_total": line_total})
    fee = BIZ["fee"] if delivery else 0.0
    result = {"lines": lines, "subtotal": round(subtotal, 2), "delivery_fee": fee, "total": round(subtotal + fee, 2)}
    QUOTE.update(result=result, shown_to_customer=False)
    return result


def place_order(items, delivery, customer_name, address=""):
    quote = QUOTE["result"]
    if quote is None or not QUOTE["shown_to_customer"]:
        return {"error": "Not placed. First show the customer the order summary and total, then wait for them to confirm."}
    priced = calculate_order_total(items, delivery)
    if "error" in priced:
        return priced
    if priced["total"] != quote["total"]:
        return {"error": "Not placed. The order changed since the customer saw the total. Show the new total and ask again."}
    QUOTE["shown_to_customer"] = True
    order = {"order_id": datetime.now().strftime("%H%M%S"), "customer_name": customer_name,
             "delivery": delivery, "address": address, **priced}
    with open(ORDERS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(order) + "\n")
    return {"status": "placed", "order_id": order["order_id"], "total": order["total"]}


def tools_schema():
    sizes = sorted({s for its in MENU.values() for ss in its.values() for s in ss})
    item = {"type": "object", "properties": {
        "name": {"type": "string", "description": "The item name exactly as on the menu"},
        "size": {"type": "string", "description": "The size exactly as on the menu: " + ", ".join(sizes)},
        "quantity": {"type": "integer"},
        "add_ons": {"type": "array", "items": {"type": "string"}, "description": "Add-ons for this item, if any"}},
        "required": ["name", "size", "quantity"]}
    return [
        {"type": "function", "function": {"name": "get_menu",
         "description": f"Get the {BIZ['type']} menu with all items, sizes, add-ons, and prices.",
         "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {"name": "calculate_order_total",
         "description": "Calculate the exact price of an order. Always use this instead of doing math yourself.",
         "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": item},
                        "delivery": {"type": "boolean", "description": "True if the order is for delivery"}},
                        "required": ["items", "delivery"]}}},
        {"type": "function", "function": {"name": "place_order",
         "description": "Place the order. Only call this after the customer has confirmed the summary and total.",
         "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": item},
                        "delivery": {"type": "boolean"}, "customer_name": {"type": "string"},
                        "address": {"type": "string", "description": "The delivery address, if delivery"}},
                        "required": ["items", "delivery", "customer_name"]}}},
    ]


FUNCTIONS = {"get_menu": get_menu, "calculate_order_total": calculate_order_total, "place_order": place_order}


def keep_final_copy(text):
    lines = (text or "").strip().split("\n")
    for k in range(1, len(lines)):
        a, b = "\n".join(lines[:k]).strip(), "\n".join(lines[k:]).strip()
        if a and b and SequenceMatcher(None, a, b).ratio() > 0.8:
            return b
    return text


def run_turn(messages):
    """The agent loop: the model asks for tools, our code runs them, until the model replies with text."""
    global client
    client = client or OpenAI()
    tools = tools_schema()
    while True:
        msg = client.chat.completions.create(model=MODEL, messages=messages, tools=tools).choices[0].message
        if not msg.tool_calls:
            reply = keep_final_copy(msg.content)
            messages.append({"role": "assistant", "content": reply})
            if QUOTE["result"] is not None:
                QUOTE["shown_to_customer"] = True
            return reply
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            try:
                result = FUNCTIONS[call.function.name](**args)
            except Exception as e:
                result = {"error": f"The tool could not run: {e}"}
            tool_log.append(f"{call.function.name}({json.dumps(args)}) -> {json.dumps(result)}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})


def not_ready():
    problems, _ = check_business()
    if problems:
        return "❌ Your business has problems to fix first. Go back to tab 1 and click **Check and save**."
    if not os.getenv("OPENAI_API_KEY"):
        return "❌ No OpenAI API key. Put OPENAI_API_KEY in the .env file in the AI500 folder, then restart the app."
    return None


# ════════════════════════════════════════════════════════════════════════════
# The app's buttons
# ════════════════════════════════════════════════════════════════════════════
def on_check(name, btype, instructions, menu, add_ons, fee):
    DATA.update(business_name=name, business_type=btype, instructions=instructions,
                menu=rows(menu), add_ons=rows(add_ons), delivery_fee=fee)
    problems, warnings = check_business()
    save()
    return status_markdown(problems, warnings)


def order_items(table):
    items, bad = [], []
    for r in rows(table):
        r += ["", "", "", ""]
        if not r[0].isdigit() or not r[1]:
            bad.append(f"Row '{' | '.join(x for x in r[:4] if x)}': write a quantity (a number) and an item name.")
            continue
        items.append({"name": r[1], "size": r[2], "quantity": int(r[0]),
                      "add_ons": [a.strip() for a in r[3].split(",") if a.strip()]})
    return items, bad


def on_price(table, hand_math):
    DATA.update(test_order=rows(table), hand_math=hand_math); save()
    problems, _ = check_business()
    if problems:
        return "❌ Your business has problems to fix first. Go back to tab 1 and click **Check and save**."
    items, bad = order_items(table)
    if bad:
        return "❌ " + "\n\n❌ ".join(bad)
    r = calculate_order_total(items, delivery=BIZ["offers_delivery"])
    if "error" in r:
        out = f"### The tool returned an error\n\n> {r['error']}\n\nCheck the spelling of your items, sizes and add-ons against your menu."
    else:
        out = ("### The tool's price\n\n| Item | Price |\n|---|---|\n" +
               "\n".join(f"| {l['item']} | ${l['line_total']:.2f} |" for l in r["lines"]) +
               f"\n| Subtotal | ${r['subtotal']:.2f} |" +
               (f"\n| Delivery fee | ${r['delivery_fee']:.2f} |" if BIZ["offers_delivery"] else "") +
               f"\n| **Total** | **${r['total']:.2f}** |\n\nNow work out the total by hand below. Does it match?")
    RESULTS["price_test"] = out
    return out


def on_skip(table):
    if not_ready():
        return not_ready()
    items, bad = order_items(table)
    QUOTE.update(result=None, shown_to_customer=False)
    r = place_order(items, delivery=False, customer_name="Test")
    out = (f"### place_order was called before any customer saw a total\n\n> **Our code answered:** {r.get('error', r)}\n\n"
           "The order was **not** saved. This rule is in the code, so the model cannot skip it, even if it tries.")
    RESULTS["skip_test"] = out
    return out


def on_run_tests(t1, t2, t3, t4, notes, progress=gr.Progress()):
    DATA.update(tests=[t1, t2, t3, t4], test_notes=notes); save()
    problem = not_ready()
    if problem:
        return problem
    parts = []
    for k, (title, text) in enumerate(zip(TEST_TITLES, [t1, t2, t3, t4])):
        progress(k / 4, desc=f"Running {title.split(':')[0]}...")
        msgs = [m.strip() for m in text.splitlines() if m.strip()]
        if not msgs:
            parts.append(f"## {title}\n\n(no customer messages)"); continue
        QUOTE.update(result=None, shown_to_customer=False)
        start = len(tool_log)
        convo = [{"role": "system", "content": BIZ["system_prompt"]}]
        lines = [f"## {title}"]
        for m in msgs:
            convo.append({"role": "user", "content": m})
            try:
                reply = run_turn(convo)
            except Exception as e:
                reply = f"⚠️ {type(e).__name__}: {e}"
            lines.append(f"**Customer:** {m}\n\n**Bot:** {reply}")
        log = tool_log[start:]
        lines.append("**Tool log** (what the bot asked our code to do):\n\n```\n" +
                     ("\n".join(l[:400] for l in log) or "(no tools called)") + "\n```")
        parts.append("\n\n".join(lines))
    out = "\n\n---\n\n".join(parts)
    RESULTS["tests"] = out
    return out


def on_chat(user_message, chat_history, conversation):
    if not user_message.strip():
        return "", chat_history, conversation, gr.update()
    problem = not_ready()
    if problem:
        return user_message, chat_history + [{"role": "assistant", "content": problem}], conversation, ""
    if not conversation:
        QUOTE.update(result=None, shown_to_customer=False)
        conversation = [{"role": "system", "content": BIZ["system_prompt"]}]
        RESULTS["chats"].append({"messages": [], "tools": []})
    conversation.append({"role": "user", "content": user_message})
    start = len(tool_log)
    try:
        reply = run_turn(conversation)
    except Exception as e:
        conversation.pop()
        return user_message, chat_history + [{"role": "assistant", "content": f"⚠️ {type(e).__name__}: {e}"}], conversation, ""
    new = tool_log[start:]
    chat = RESULTS["chats"][-1]
    chat["messages"] += [("Customer", user_message), ("Bot", reply)]
    chat["tools"] += new
    chat_history = chat_history + [{"role": "user", "content": user_message}, {"role": "assistant", "content": reply}]
    return "", chat_history, conversation, "\n\n".join(new) or "(no tools used this turn)"


def on_notes(text):
    DATA["test_notes"] = text
    save()


def on_new_order():
    QUOTE.update(result=None, shown_to_customer=False)
    return "", [], [], ""


def orders_markdown():
    if not ORDERS_FILE.exists():
        return "No orders saved yet."
    out = ["| Order | Customer | Total | Pickup or delivery | Items |", "|---|---|---|---|---|"]
    for line in ORDERS_FILE.read_text(encoding="utf-8").splitlines():
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        out.append(f"| #{o['order_id']} | {o['customer_name']} | ${o['total']:.2f} | "
                   + ("delivery to " + o["address"] if o["delivery"] else "pickup") + " | "
                   + "; ".join(l["item"] for l in o["lines"]) + " |")
    return "\n".join(out)


def on_clear_orders():
    if ORDERS_FILE.exists():
        ORDERS_FILE.unlink()
    return "No orders saved yet."


def on_report(*answers):
    DATA["checklist"] = [[answers[2 * i], answers[2 * i + 1]] for i in range(6)]; save()
    check_business()
    e = lambda s: html.escape(str(s))
    def md_block(s):
        if not s:
            return "<p><i>Not done yet.</i></p>"
        try:                                       # markdown -> HTML (markdown-it-py comes with Gradio)
            from markdown_it import MarkdownIt
            return MarkdownIt("commonmark", {"html": False}).enable("table").render(str(s))
        except Exception:
            return f"<pre>{e(s)}</pre>"
    chats = ""
    for k, c in enumerate([c for c in RESULTS["chats"] if c["messages"]], 1):
        chats += f"<h3>Chat {k}</h3>" + "".join(f"<p><b>{who}:</b> {e(t)}</p>" for who, t in c["messages"])
        chats += "<p><b>Tool log:</b></p><pre>" + e("\n".join(c["tools"]) or "(no tools called)") + "</pre>"
    menu_rows = "".join(f"<tr><td>{e(c)}</td><td>{e(i)}</td><td>{e(', '.join(f'{s} ${p:.2f}' for s, p in z.items()))}</td></tr>"
                        for c, its in MENU.items() for i, z in its.items())
    check_rows = "".join(f"<tr><td>{e(q)}</td><td>{e(a[0] or '-')}</td><td>{e(a[1])}</td></tr>"
                         for q, a in zip(CHECK_QUESTIONS, DATA["checklist"]))
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(BIZ['name'])} OrderBot</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:900px;margin:24px auto;padding:0 16px;line-height:1.45}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}
pre{{background:#f5f5f5;padding:10px;white-space:pre-wrap}}h2{{border-bottom:2px solid #333;padding-bottom:4px}}</style></head><body>
<h1>{e(BIZ['name'])} OrderBot: my results</h1><p>Created {datetime.now():%Y-%m-%d %H:%M} with the OrderBot Builder ({MODEL}).</p>
<h2>Part 1. My business</h2><p><b>Business:</b> {e(BIZ['name'])} ({e(BIZ['type'])})</p>
<table><tr><th>Category</th><th>Item</th><th>Sizes and prices</th></tr>{menu_rows}</table>
<p><b>Add-ons:</b> {e(', '.join(add_on_text(a) for a in ADD_ONS) or 'none')}<br>
<b>Delivery:</b> {e(f"${BIZ['fee']:.2f} fee" if BIZ['offers_delivery'] else 'pickup only')}</p>
<h2>Part 2. My assistant's instructions</h2><pre>{e(DATA['instructions'])}</pre>
<h2>Part 3. Test a tool</h2>{md_block(RESULTS['price_test'])}<p><b>My hand calculation:</b></p>{md_block(DATA.get('hand_math'))}
{md_block(RESULTS['skip_test'])}
<h2>Part 4. Test conversations</h2>{md_block(RESULTS['tests'])}<p><b>My notes on the tests:</b></p>{md_block(DATA.get('test_notes'))}
<h2>Part 5. Chats in the app</h2>{chats or '<p><i>No chats yet.</i></p>'}
<h3>Saved orders</h3>{md_block(orders_markdown())}
<h2>Testing checklist</h2><table><tr><th>Question</th><th>Result</th><th>Evidence</th></tr>{check_rows}</table>
</body></html>"""
    REPORT_FILE.write_text(page, encoding="utf-8")
    missing = [n for n, ok in (("the tool test (tab 2)", RESULTS["price_test"]), ("the four test conversations (tab 3)", RESULTS["tests"]),
                               ("a chat in tab 4", any(c["messages"] for c in RESULTS["chats"])),
                               ("the checklist", all(a[0] for a in DATA["checklist"]))) if not ok]
    note = ("✅ Your results file is ready. Download it below and submit it with your report."
            if not missing else "⚠️ The file is ready, but it is missing: " + ", ".join(missing) +
            ". Results are kept only while the app is running, so do those steps, then click this button again.")
    return note, str(REPORT_FILE)


# ════════════════════════════════════════════════════════════════════════════
# The screen
# ════════════════════════════════════════════════════════════════════════════
def build():
    load_saved()
    with gr.Blocks(title="OrderBot Builder") as demo:
        gr.Markdown("# 🛎️ OrderBot Builder\nBuild an ordering chatbot for your own business, test it, and chat with it. "
                    "Work through the tabs from left to right. Everything you type is saved as you go.")
        with gr.Tab("1. Your business"):
            gr.Markdown("Describe your business and its menu. Replace the example bagel shop with your own (you can make "
                        "the business up). You need **at least 6 items in at least 2 categories**, **one item with more "
                        "than one size**, and **at least 3 add-ons**.")
            with gr.Row():
                name = gr.Textbox(DATA["business_name"], label="Business name")
                btype = gr.Textbox(DATA["business_type"], label="Type of business (for example: coffee shop)")
                fee = gr.Textbox(DATA["delivery_fee"], label="Delivery fee (or none for pickup only)")
            instructions = gr.Textbox(DATA["instructions"], lines=5, label="Instructions for your assistant: its name and "
                                      "tone, and facts not on the menu (hours, pickup time, delivery area, policies)")
            gr.Markdown("**Menu.** One item per row. In *Sizes and prices*, write each size and its price, separated by "
                        "commas (`small 2.00, large 3.00`). For an item with one size, write just the price (`2.50`). "
                        "Click a cell to type in it; use the empty rows for more items. Rows you leave empty are ignored.")
            menu = gr.Dataframe(padded(DATA["menu"], 3, 15), headers=["Category", "Item", "Sizes and prices"], datatype="str",
                                column_count=3, interactive=True, label="Menu")
            add_ons = gr.Dataframe(padded([add_on_row(r) for r in DATA["add_ons"]], 3, 8), headers=["Add-on", "Category", "Extra price"], datatype="str",
                                   column_count=3, interactive=True,
                                   label="Add-ons: extras a customer can add to an item, like oat milk or extra cheese. Category: the menu category it can be added to (for example, drinks); leave it empty if it can be added to any item.")
            check_btn = gr.Button("✅ Check and save", variant="primary")
            status = gr.Markdown()
            check_btn.click(on_check, [name, btype, instructions, menu, add_ons, fee], status)

        with gr.Tab("2. Test a tool"):
            gr.Markdown("Before any AI is involved, test the **calculate_order_total** tool with an order from your menu: "
                        "**at least 3 items, using add-ons**. Leave *Size* empty for an item with one size; separate "
                        "several add-ons with commas.")
            order = gr.Dataframe(padded(DATA["test_order"], 4, 8), headers=["Quantity", "Item", "Size", "Add-ons"], datatype="str",
                                 column_count=4, interactive=True, label="Test order")
            price_btn = gr.Button("🧮 Price this order", variant="primary")
            price_out = gr.Markdown()
            hand = gr.Textbox(DATA.get("hand_math", ""), lines=3, label="Your answer: work out the total by hand "
                              "(show the math). Does it match the tool's total?")
            price_btn.click(on_price, [order, hand], price_out)
            gr.Markdown("Now try to **skip the confirmation**: call place_order before any customer has seen a total.")
            skip_btn = gr.Button("🚫 Try to place this order without confirmation")
            skip_out = gr.Markdown()
            skip_btn.click(on_skip, [order], skip_out)

        with gr.Tab("3. Test conversations"):
            gr.Markdown("Rewrite each test for your business: **one customer message per line**. Keep the purpose of "
                        "each test. Then run them and read each conversation and its **tool log**. (About a minute.)")
            boxes = []
            with gr.Row():
                for k in (0, 1):
                    boxes.append(gr.Textbox(DATA["tests"][k], lines=4, label=TEST_TITLES[k]))
            with gr.Row():
                for k in (2, 3):
                    boxes.append(gr.Textbox(DATA["tests"][k], lines=4, label=TEST_TITLES[k]))
            run_btn = gr.Button("▶️ Run the four tests", variant="primary")
            tests_out = gr.Markdown()
            notes = gr.Textbox(DATA.get("test_notes", ""), lines=4, label="Your answer: did the bot behave as you "
                               "expected in each test? Which tools did it call, and when? If something went wrong, "
                               "what did you change, and did it fix the problem?")
            run_btn.click(on_run_tests, boxes + [notes], tests_out)
            notes.blur(on_notes, notes, None)

        with gr.Tab("4. Chat with your bot"):
            gr.Markdown("Place at least **one full order**, and try to break the bot: ask for something you do not "
                        "sell, change your mind, or ask for the total before you finish. On the right, watch which "
                        "tools the bot calls. Take a screenshot of a conversation and its tool panel for your report.")
            conversation = gr.State([])
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=430, label="Your bot")
                    msg = gr.Textbox(placeholder="Type your message and press Enter", label="Your message")
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        new = gr.Button("New order")
                with gr.Column(scale=2):
                    tools_box = gr.Textbox(label="Tool calls this turn", lines=19, interactive=False)
            msg.submit(on_chat, [msg, chatbot, conversation], [msg, chatbot, conversation, tools_box])
            send.click(on_chat, [msg, chatbot, conversation], [msg, chatbot, conversation, tools_box])
            new.click(on_new_order, None, [msg, chatbot, conversation, tools_box])
            gr.Markdown("**Orders your bot saved**")
            orders = gr.Markdown(orders_markdown())
            with gr.Row():
                gr.Button("🔄 Refresh the order list").click(orders_markdown, None, orders)
                gr.Button("🗑️ Clear the order list").click(on_clear_orders, None, orders)

        with gr.Tab("5. Checklist and submit"):
            gr.Markdown("Fill in the testing checklist, using your test conversations and at least one full order in "
                        "tab 4. Then create your results file and submit it with your Word report.")
            answers = []
            for q, a in zip(CHECK_QUESTIONS, DATA["checklist"]):
                with gr.Row():
                    answers.append(gr.Radio(["Yes", "No", "Partly"], value=a[0] or None, label=q, scale=2))
                    answers.append(gr.Textbox(a[1], label="Evidence (for example: which test shows it)", scale=3))
            report_btn = gr.Button("📄 Create my results file", variant="primary")
            report_note = gr.Markdown()
            report_file = gr.File(label="Your results file (download it and submit it)")
            report_btn.click(on_report, answers, [report_note, report_file])

        with gr.Tab("Quit"):
            gr.Markdown("When you are done, click **Quit** to stop the app and its public link. Your business and "
                        "answers are saved; test results and chats are not, so create your results file first.")
            quit_btn = gr.Button("⏹ Quit the app", variant="stop")
            quit_note = gr.Markdown()
            quit_btn.click(lambda: (threading.Timer(1, demo.close).start(), "The app has stopped. You can close this tab.")[1],
                           None, quit_note)
    return demo


def launch(share=True):
    """Start the app. On the HPC server, open the https://....gradio.live link it prints."""
    demo = build()          # build() loads the saved file
    check_business()
    demo.launch(share=share)
    return demo
