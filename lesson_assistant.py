"""
Lesson Assistant — Phase 1
----------------------------
Made by Bilal Ahmad

Topic do -> Urdu explanation + Quiz (MCQs with answers) + optional YouTube script.

Run:
    streamlit run lesson_assistant.py

API key ab Streamlit "Secrets" se aati hai (users ko dikhti nahi).
Local mein test karne ke liye .streamlit/secrets.toml mein daalein:
    GEMINI_API_KEY = "your-key-here"
"""

import json
import datetime
from pathlib import Path

import streamlit as st
import google.generativeai as genai


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

MODEL_NAME = "gemini-3.6-flash"   # switch karna ho to yahan model string badlo
MAX_TOKENS = 2000

# Poora din mein total kitni "generations" allowed hain (sab users milakar).
# Ek "generation" = ek Generate click (explanation + quiz + optional script).
# Ye limit khud-ba-khud agle din reset ho jati hai, kisi manual kaam ki
# zaroorat nahi.
DAILY_GENERATION_LIMIT = 50

USAGE_FILE = Path(__file__).parent / ".usage_counter.json"


# ---------------------------------------------------------------------------
# DAILY USAGE LIMIT
# Server-side counter (file par based) — session state ya browser se
# bypass nahi ho sakta, kyunke ye disk par store hota hai, user ke paas
# nahi. Date badalte hi khud reset ho jata hai.
# ---------------------------------------------------------------------------

def _read_usage() -> dict:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"date": "", "count": 0}


def _write_usage(data: dict) -> None:
    try:
        USAGE_FILE.write_text(json.dumps(data))
    except OSError:
        pass  # agar disk write fail ho, silently ignore — limit fail-open rahega


def get_remaining_generations() -> int:
    today = datetime.date.today().isoformat()
    data = _read_usage()
    if data.get("date") != today:
        return DAILY_GENERATION_LIMIT
    return max(0, DAILY_GENERATION_LIMIT - data.get("count", 0))


def record_generation() -> None:
    today = datetime.date.today().isoformat()
    data = _read_usage()
    if data.get("date") != today:
        data = {"date": today, "count": 0}
    data["count"] += 1
    _write_usage(data)


# ---------------------------------------------------------------------------
# PROMPT BUILDERS
# Ye teeno functions asal "product" hain. Yahin se output ka quality,
# tone, aur structure control hota hai. Baaki sab sirf UI/plumbing hai.
# ---------------------------------------------------------------------------

def build_explanation_prompt(topic: str) -> str:
    """
    Urdu mein simple, saaf, student-friendly explanation ke liye prompt.
    """
    return f"""Aap ek tajurbakar Urdu-medium teacher hain jo mushkil topics ko
aasan aur rozmarra ki misalon se samjhate hain.

Topic: "{topic}"

Neeche di gayi requirements ke mutabiq is topic ki explanation likhein:

1. Sirf Urdu (Nastaliq script) mein likhein — koi Roman Urdu ya English nahi.
2. Ek chhoti si intro se shuru karein ke ye topic kyun important hai.
3. Asal concept ko 3-5 chhote paragraphs mein samjhayein.
4. Kam az kam ek real-zindagi ki misal shamil karein.
5. Mushkil alfaz istemal na karein; jahan zaroori ho wahan aasan tashreeh dein.
6. Aakhir mein 2-3 jumlon ka khulasa (summary) dein.
7. Total length: taqreeban 250-400 alfaz.

Sirf explanation likhein, koi extra heading ya meta-commentary nahi."""


def build_quiz_prompt(topic: str, explanation: str = "") -> str:
    """
    5 MCQs, har ek ke 4 options aur sahih jawab ke saath.
    Agar explanation available ho to usi content se sawal banayein taake
    quiz aur lesson aapas mein consistent rahein.
    """
    context_block = (
        f"\n\nYe raha wo explanation jo student ko diya gaya hai — isi ke\n"
        f"content se sawal banayein:\n\"\"\"\n{explanation}\n\"\"\""
        if explanation else ""
    )

    return f"""Aap ek Urdu-medium teacher hain jo assessment quizzes banate hain.

Topic: "{topic}"{context_block}

5 multiple-choice sawal (MCQs) banayein jo is topic ki bunyadi samajh test karein.

Format strictly is tarah follow karein (Urdu mein):

سوال 1: [سوال کا متن]
الف) [آپشن]
ب) [آپشن]
ج) [آپشن]
د) [آپشن]
صحیح جواب: [حرف] — [ایک سطر کی مختصر وضاحت]

Requirements:
1. Sawalon ki difficulty easy se medium tak honi chahiye (beginner student ke liye).
2. Har sawal ka sirf ek sahih jawab ho, baaki 3 plausible lekin ghalat options hon.
3. Sawal topic ke different pehluon ko cover karein, ek hi cheez baar baar na poochein.
4. Sirf Urdu mein likhein.
5. Sirf 5 sawal do, koi intro ya extra text nahi."""


def build_script_prompt(topic: str, explanation: str = "") -> str:
    """
    Optional: chhoti YouTube video ke liye script (5-7 minute ka talk-through).
    """
    context_block = (
        f"\n\nReference ke liye, ye explanation pehle di ja chuki hai:\n"
        f"\"\"\"\n{explanation}\n\"\"\""
        if explanation else ""
    )

    return f"""Aap ek YouTube educator hain jo Urdu-medium students ke liye
5-7 minute ki short educational videos banate hain.

Topic: "{topic}"{context_block}

Is topic par ek YouTube video script likhein jo bolne ke liye ready ho
(spoken Urdu, thoda informal aur engaging lehja).

Script mein ye sections hon:

1. HOOK (10-15 seconds): Ek dilchasp sawal ya fact se shuru karein taake
   viewer ruk kar dekhe.
2. INTRO (15-20 seconds): Bataein aaj kya seekhenge aur ye kyun useful hai.
3. MAIN CONTENT (3-4 minutes): Topic ko step-by-step, misalon ke saath
   samjhayein — jaise camera ke saamne bol rahe hon.
4. QUICK RECAP (30 seconds): 2-3 bullet points mein khulasa.
5. OUTRO (10-15 seconds): Viewer ko like/subscribe/comment ke liye
   encourage karein aur agla topic tease karein.

Har section ko clearly label karein. Bolne wale lehje mein likhein
(short sentences, natural pauses), na ke essay ki tarah."""


# ---------------------------------------------------------------------------
# GEMINI API CALL
# ---------------------------------------------------------------------------

def call_claude(api_key: str, prompt: str) -> str:
    """
    Naam 'call_claude' hi rakha hai taake baaki app code (jo is function ko
    call karta hai) bina badle chal sake. Andar se ye ab Gemini use karta hai.
    Kisi aur provider par switch karna ho to sirf yahi function badalna hoga.
    """
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=MAX_TOKENS,
        ),
    )
    return response.text


# ---------------------------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Lesson Assistant", page_icon="📘", layout="centered")
    st.title("📘 Lesson Assistant — Phase 1")
    st.caption("Topic do → Urdu explanation + Quiz + optional YouTube script")

    # ---- Sidebar: options (API key ab secrets se aati hai, user ko nahi dikhti) ----
    with st.sidebar:
        st.header("Settings")
        remaining = get_remaining_generations()
        st.metric("Aaj baaki generations", f"{remaining} / {DAILY_GENERATION_LIMIT}")
        st.divider()
        generate_script = st.checkbox("YouTube script bhi banao", value=False)
        st.divider()
        st.markdown(
            "**Note:** Ye Phase 1 hai, Gemini API use ho rahi hai. Model switch "
            "karna ho (OpenAI/Claude) to `call_claude` function replace karein — "
            "baaki structure same rahega."
        )

    # ---- Main input ----
    topic = st.text_input("Topic likhein (kisi bhi zaban mein)", placeholder="Misal: Photosynthesis")
    generate_btn = st.button("Generate", type="primary", use_container_width=True)

    if generate_btn:
        if get_remaining_generations() <= 0:
            st.error(
                "Aaj ka generation limit khatam ho chuka hai. Kal phir try karein "
                "(limit har roz khud reset ho jati hai)."
            )
            return
        if not topic.strip():
            st.error("Pehle koi topic likhein.")
            return

        api_key = ""
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass  # secrets.toml na ho to bhi app crash na ho, sirf error dikhaye
        if not api_key:
            st.error(
                "API key configure nahi hai. App owner ko Streamlit Secrets mein "
                "GEMINI_API_KEY set karna hoga."
            )
            return

        # 1) Explanation
        with st.spinner("Urdu explanation ban rahi hai..."):
            try:
                explanation = call_claude(api_key, build_explanation_prompt(topic))
            except Exception as e:
                st.error(f"Explanation generate karte waqt error aaya: {e}")
                return

        st.session_state["explanation"] = explanation
        st.session_state["topic"] = topic

        # 2) Quiz
        with st.spinner("Quiz ban raha hai..."):
            try:
                quiz = call_claude(api_key, build_quiz_prompt(topic, explanation))
            except Exception as e:
                st.error(f"Quiz generate karte waqt error aaya: {e}")
                return

        st.session_state["quiz"] = quiz

        # 3) Optional script
        if generate_script:
            with st.spinner("YouTube script ban raha hai..."):
                try:
                    script = call_claude(api_key, build_script_prompt(topic, explanation))
                    st.session_state["script"] = script
                except Exception as e:
                    st.error(f"Script generate karte waqt error aaya: {e}")
        else:
            st.session_state.pop("script", None)

        # Generation successful hone ke baad hi count badhta hai — taake
        # failed attempts users ka quota waste na karein.
        record_generation()

    # ---- Display results ----
    if "explanation" in st.session_state:
        st.subheader("📖 Urdu Explanation")
        st.markdown(st.session_state["explanation"])

        st.subheader("📝 Quiz")
        st.markdown(st.session_state["quiz"])

        if "script" in st.session_state:
            st.subheader("🎬 YouTube Script")
            st.markdown(st.session_state["script"])

    # ---- Footer / credit ----
    st.divider()
    st.caption("Made by Bilal Ahmad")


if __name__ == "__main__":
    main()
