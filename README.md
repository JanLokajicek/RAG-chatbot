# RAG Chatbot — pojistné dokumenty

Streamlit chatbot pro dotazování nad PDF pojistnými dokumenty.
Používá Claude (Anthropic) jako LLM a lokální sentence-transformers embeddingy.

## Spuštění lokálně

### 1. Závislosti

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. API klíč

Vytvoř soubor `.env` v kořeni projektu:

```
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Spuštění

```bash
streamlit run app.py
```

Aplikace běží na `http://localhost:8501`.

---

## Deploy na Streamlit Cloud

1. Pushni repozitář na GitHub (`.env` a `secrets.toml` jsou v `.gitignore`)
2. Na [share.streamlit.io](https://share.streamlit.io) propoj GitHub repozitář
3. V nastavení aplikace → **Secrets** vlož:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Klikni **Deploy**

---

## Struktura projektu

```
app.py              # Streamlit UI
rag_app.py          # RAG pipeline (embeddingy, retriever, LLM)
chainlit_app.py     # Alternativní Chainlit UI
requirements.txt    # Závislosti
.env                # API klíč (lokálně, není v gitu)
.streamlit/
  secrets.toml      # API klíč pro Streamlit Cloud (není v gitu)
uploaded_docs/      # Nahrané PDF soubory (není v gitu)
chroma_db/          # Vektorová databáze (není v gitu)
evaluations.csv     # Hodnocení odpovědí (není v gitu)
```

## Funkce

- Nahrávání více PDF dokumentů přes sidebar
- Hybridní vyhledávání (sémantické + BM25)
- Automatická analýza typů pojištění při startu
- Hodnocení odpovědí (👍/👎) s exportem do CSV
- Měření latence odpovědí
- Perzistence dokumentů mezi restarty
