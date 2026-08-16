# Lesson Assistant — Phase 1

**Made by Bilal Ahmad**

Topic do, teen cheezein milti hain:

1. **Urdu explanation** — simple, student-friendly
2. **Quiz** — 5 MCQs jawabon ke saath
3. **YouTube script** (optional) — 5-7 minute ki spoken video script

Gemini API use karta hai (free tier). Structure model-agnostic hai — `call_claude`
function replace karke OpenAI ya Anthropic mein switch kiya ja sakta hai.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run lesson_assistant.py
```

Browser mein app khul jayega (default: `http://localhost:8501`).

## API Key (ab hidden hai)

Users ko ab koi API key box nahi dikhta. Key Streamlit "Secrets" mein
chhupi hoti hai — sirf app owner (aap) set karte hain.

**Local testing ke liye:**
1. `.streamlit/secrets.toml.example` ko copy karke `.streamlit/secrets.toml`
   bana lein (same folder mein).
2. Usme apni free Gemini key daal dein: https://aistudio.google.com/apikey
3. `secrets.toml` ko kabhi GitHub par push na karein — `.gitignore` mein
   already excluded hai.

**Streamlit Community Cloud par deploy karte waqt:**
App settings mein "Secrets" section hota hai — wahan yehi `GEMINI_API_KEY`
line paste kar dein. Users ko kuch dikhega nahi, seedha topic type karke
Generate dabayenge.

## Daily generation limit

Kyunke key ab shared hai (sab users ek hi key use karte hain), app mein
ek built-in daily cap hai — `lesson_assistant.py` ke top par:
```python
DAILY_GENERATION_LIMIT = 50
```
Ye limit khud reset hoti hai har naye din. Counter ek chhoti si file
(`.usage_counter.json`) mein server-side store hota hai — users ise
dekh ya badal nahi sakte, is liye khud bypass nahi ho sakta. Limit
badhani/ghatani ho to bas ye number badal dein.

**Note:** Streamlit Community Cloud ka free tier ek single instance
chalata hai, is liye file-based counter theek se kaam karta hai. Agar
kabhi app multiple servers par scale ho (bahut zyada traffic), to counter
ko database (jaise Firebase/Supabase) mein move karna behtar hoga.

## Files

| File | Kaam |
|---|---|
| `lesson_assistant.py` | Main app — UI + Gemini API calls + prompts |
| `requirements.txt` | Dependencies (streamlit, google-generativeai) |
| `README.md` | Ye file |

## Prompts kahan hain?

`lesson_assistant.py` mein teen functions hain jo asal "product" hain:

- `build_explanation_prompt(topic)`
- `build_quiz_prompt(topic, explanation)`
- `build_script_prompt(topic, explanation)`

Output ka quality ya tone improve karna ho (behtar quiz, mukhtalif lehja, etc.),
to yahi functions edit karein — UI code ko chhedne ki zaroorat nahi.

## Model badalna

Model string `lesson_assistant.py` ke top par `MODEL_NAME` variable mein hai
(abhi `gemini-2.5-flash`). Gemini ke alawa doosri API (OpenAI/Anthropic) use
karni ho to sirf `call_claude()` function replace karein — baaki poora
structure (prompts, UI, session state) waisa hi rahega.

## Next steps (Phase 2 ideas)

- Multiple topics ek saath (batch mode)
- Quiz ko interactive banana (score checking UI mein)
- Generated content ko file mein save/export karna
- Different difficulty levels (beginner/intermediate/advanced)
