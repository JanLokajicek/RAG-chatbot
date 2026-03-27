"""
RAG (Retrieval-Augmented Generation) aplikace
==============================================
Načte PDF, vytvoří embeddingy, uloží do ChromaDB a odpovídá na otázky.
LLM: Claude (Anthropic) | Embeddingy: sentence-transformers (lokálně, zdarma)

==============================================================================
ČISTÝ KÓD BEZ KOMENTÁŘŮ — projdi si ho nejdřív, pak viz živý kód níže
==============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PDF_PATH = "document.pdf"
CHROMA_DIR = "./chroma_db"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 400
TOP_K = 10

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()

def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\\n\\n", "\\n", ". ", " ", ""],
    )
    return splitter.split_documents(pages)

def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
    )

def build_vectorstore(chunks, persist_dir):
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embedding_model(),
        persist_directory=persist_dir,
    )

def load_vectorstore(persist_dir):
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embedding_model(),
    )

def build_rag_chain(vectorstore, chunks):
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K
    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Jsi pomocný asistent analyzující pojistný dokument.\\n\\n"
            "INSTRUKCE:\\n"
            "1. Přečti si pozorně celý kontext níže.\\n"
            "2. Pokud kontext obsahuje odpověď na otázku — odpověz podrobně a uveď všechny relevantní body.\\n"
            "3. Pokud kontext odpověď VŮBEC neobsahuje — napiš pouze větu: 'Tuto informaci dokument neobsahuje.'\\n"
            "4. NIKDY nepište obě věci najednou. Buď odpovíš z kontextu, nebo napíšeš že informace chybí.\\n"
            "5. Nepřidávej žádné domněnky ani informace které nejsou v kontextu.\\n\\n"
            "Kontext:\\n{context}\\n\\n"
            "Otázka: {question}\\n\\n"
            "Odpověď:"
        ),
    )
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",
        temperature=0,
        anthropic_api_key=ANTHROPIC_API_KEY,
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt_template},
    )

def interactive_loop(chain):
    print("=" * 60)
    print("RAG chatbot připraven! Piš otázky (quit = konec).")
    print("=" * 60)
    while True:
        question = input("\\nOtázka: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "konec"):
            break
        result = chain.invoke({"query": question})
        print(f"\\nOdpověď:\\n{result['result']}")
        sources = result.get("source_documents", [])
        if sources:
            pages = sorted({doc.metadata.get("page", "?") + 1 for doc in sources})
            print(f"\\nZdroje — stránky PDF: {pages}")

def main(pdf_path=PDF_PATH):
    pages = load_pdf(pdf_path)
    chunks = split_documents(pages)
    chroma_exists = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())
    if chroma_exists:
        vectorstore = load_vectorstore(CHROMA_DIR)
    else:
        vectorstore = build_vectorstore(chunks, CHROMA_DIR)
    chain = build_rag_chain(vectorstore, chunks)
    interactive_loop(chain)

if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else PDF_PATH)

==============================================================================
ŽIVÝ KÓD S KOMENTÁŘI — viz níže
==============================================================================
"""

import os                          # přístup k proměnným prostředí (API klíče)
from pathlib import Path           # práce s cestami k souborům a složkám

from dotenv import load_dotenv     # načte .env soubor do os.environ
load_dotenv()                      # spustí načtení — musí být před os.getenv()

from langchain_community.document_loaders import PyPDFLoader          # načítání PDF souborů
from langchain_text_splitters import RecursiveCharacterTextSplitter    # dělení textu na chunky
from langchain_community.embeddings import HuggingFaceEmbeddings      # lokální embedding model
from langchain_chroma import Chroma                                    # vektorová databáze ChromaDB
from langchain_core.prompts import PromptTemplate                     # šablona promptu pro LLM
from langchain_anthropic import ChatAnthropic                         # Claude LLM od Anthropic
from langchain_community.retrievers import BM25Retriever              # klíčové vyhledávání (BM25)
from langchain.retrievers import EnsembleRetriever                    # kombinuje více retrieverů


# ---------------------------------------------------------------------------
# KONFIGURACE
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")  # přečte klíč z .env souboru

PDF_PATH = "PP_odpovednost_obc.pdf"    # výchozí cesta k PDF (lze přepsat argumentem příkazové řádky)

CHROMA_DIR = "./chroma_db"   # složka kde ChromaDB ukládá vektory na disk

CHUNK_SIZE = 2000       # maximální počet znaků v jednom chunku
CHUNK_OVERLAP = 400     # počet znaků překryvu mezi sousedními chunky (zachová kontext na hranicích)

TOP_K = 10              # kolik nejrelevantnějších chunků se pošle do LLM jako kontext

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Jsi pomocný asistent analyzující pojistný dokument.\n\n"
        "INSTRUKCE:\n"
        "1. Přečti si pozorně celý kontext níže.\n"
        "2. Pokud kontext obsahuje odpověď na otázku — odpověz podrobně a uveď všechny relevantní body.\n"
        "3. Pokud kontext odpověď VŮBEC neobsahuje — napiš pouze větu: 'Tuto informaci dokument neobsahuje.'\n"
        "4. NIKDY nepište obě věci najednou. Buď odpovíš z kontextu, nebo napíšeš že informace chybí.\n"
        "5. Nepřidávej žádné domněnky ani informace které nejsou v kontextu.\n\n"
        "Kontext:\n{context}\n\n"
        "Otázka: {question}\n\n"
        "Odpověď:"
    ),
)


# ---------------------------------------------------------------------------
# KROK 1 — NAČTENÍ PDF
# ---------------------------------------------------------------------------

def load_pdf(path: str):
    """
    PyPDFLoader projde každou stránku PDF a vrátí seznam Document objektů.
    Každý Document má atribut .page_content (text) a .metadata (číslo stránky aj.).
    """
    print(f"\n[1] Načítám PDF: {path}")
    loader = PyPDFLoader(path)        # vytvoří loader pro daný soubor
    pages = loader.load()             # načte všechny stránky jako seznam Document objektů
    print(f"    Načteno {len(pages)} stránek.")
    return pages                      # vrátí seznam stránek pro další zpracování


# ---------------------------------------------------------------------------
# KROK 2 — ROZDĚLENÍ NA CHUNKY
# ---------------------------------------------------------------------------

def split_documents(pages):
    """
    RecursiveCharacterTextSplitter rozdělí text na menší kousky.
    Zkouší oddělovače v pořadí: odstavec → řádek → věta → slovo,
    aby chunky dávaly smysl i sémanticky.
    """
    print(f"\n[2] Dělím text na chunky (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})…")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,                              # max. délka chunku ve znacích
        chunk_overlap=CHUNK_OVERLAP,                        # překryv — konec jednoho = začátek dalšího
        separators=["\n\n", "\n", ". ", " ", ""],           # zkouší dělit postupně od největšího celku
    )
    chunks = splitter.split_documents(pages)   # rozdělí každou stránku na menší části
    print(f"    Vytvořeno {len(chunks)} chunků.")
    return chunks                              # vrátí seznam všech chunků


# ---------------------------------------------------------------------------
# KROK 3 — EMBEDDINGY A ULOŽENÍ DO CHROMADB
# ---------------------------------------------------------------------------

def get_embedding_model():
    """
    HuggingFaceEmbeddings běží lokálně — nepotřebuje žádné API ani klíče.
    Multijazyčný model vhodný pro česky psané dokumenty.
    Při prvním spuštění se automaticky stáhne (~470MB).
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # multijazyčný model
        model_kwargs={"device": "cpu"},   # spustí model na CPU (nevyžaduje GPU)
    )


def build_vectorstore(chunks, persist_dir: str):
    """
    Embedding model převede každý chunk na číselný vektor.
    Chroma tyto vektory uloží na disk — příště je načte bez přepočítávání.
    """
    print(f"\n[3] Generuji embeddingy a ukládám do ChromaDB ({persist_dir})…")
    print("    (první spuštění může stáhnout model ~470MB)")

    vectorstore = Chroma.from_documents(
        documents=chunks,                  # chunky které chceme uložit
        embedding=get_embedding_model(),   # model který převede text na vektor
        persist_directory=persist_dir,     # složka pro uložení na disk
    )

    print(f"    Uloženo {vectorstore._collection.count()} vektorů.")
    return vectorstore    # vrátí připravenou vektorovou databázi


def load_vectorstore(persist_dir: str):
    """
    Pokud databáze již existuje, načteme ji bez regenerace embeddingů.
    """
    print(f"\n[3] Načítám existující ChromaDB z {persist_dir}…")
    return Chroma(
        persist_directory=persist_dir,          # složka s uloženými vektory
        embedding_function=get_embedding_model(),  # stejný model jako při ukládání
    )


# ---------------------------------------------------------------------------
# KROK 3b — SAMOSTATNÝ RETRIEVER (pro přímé streamování)
# ---------------------------------------------------------------------------

def build_retriever(vectorstore, chunks):
    """
    Vrátí hybridní EnsembleRetriever (vektorové podobnosti + BM25).
    Používá se v app.py pro přímé volání LLM se streamingem.
    """
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K
    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )


# ---------------------------------------------------------------------------
# KROK 4 — VYTVOŘENÍ RAG ŘETĚZCE
# ---------------------------------------------------------------------------

def build_rag_chain(vectorstore, chunks):
    """
    RetrievalQA spojuje dvě části:
      • Retriever  — hybridní: sémantické vyhledávání (Chroma) + klíčová slova (BM25)
      • LLM        — Claude dostane otázku + chunky a vygeneruje odpověď

    Prompt explicitně říká modelu, aby odpovídal jen z poskytnutého kontextu.
    """
    print("\n[4] Sestavuji RAG řetězec…")

    # Sémantický retriever — hledá chunky podle vektorové podobnosti (cosine similarity)
    vector_retriever = vectorstore.as_retriever(
        search_type="similarity",       # typ vyhledávání: nejpodobnější vektory
        search_kwargs={"k": TOP_K},     # vrátí TOP_K nejpodobnějších chunků
    )

    # BM25 retriever — klasické klíčové vyhledávání, najde chunky s přesnými slovy z otázky
    bm25_retriever = BM25Retriever.from_documents(chunks)  # vytvoří index ze všech chunků
    bm25_retriever.k = TOP_K                                # vrátí TOP_K nejlepších shod

    # Hybridní retriever — sloučí výsledky obou retrieverů, každý má váhu 50 %
    retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],  # seznam retrieverů ke kombinaci
        weights=[0.5, 0.5],                             # stejná váha pro oba přístupy
    )

    # Prompt šablona — {context} se nahradí nalezenými chunky, {question} otázkou uživatele
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],   # proměnné které LangChain doplní
        template=(
            "Jsi pomocný asistent analyzující pojistný dokument.\n\n"
            "INSTRUKCE:\n"
            "1. Přečti si pozorně celý kontext níže.\n"
            "2. Pokud kontext obsahuje odpověď na otázku — odpověz podrobně a uveď všechny relevantní body.\n"
            "3. Pokud kontext odpověď VŮBEC neobsahuje — napiš pouze větu: 'Tuto informaci dokument neobsahuje.'\n"
            "4. NIKDY nepište obě věci najednou. Buď odpovíš z kontextu, nebo napíšeš že informace chybí.\n"
            "5. Nepřidávej žádné domněnky ani informace které nejsou v kontextu.\n\n"
            "Kontext:\n{context}\n\n"
            "Otázka: {question}\n\n"
            "Odpověď:"
        ),
    )

    # Claude Haiku — nejrychlejší a nejlevnější Claude model, ideální pro RAG
    llm = ChatAnthropic(
        model="claude-haiku-4-5-20251001",   # konkrétní verze modelu
        temperature=0,                        # 0 = deterministické odpovědi bez náhodnosti
        anthropic_api_key=ANTHROPIC_API_KEY, # API klíč pro autentizaci
        streaming=True,                       # token-by-token streaming pro Streamlit
    )

    # RetrievalQA řetězec — při každém dotazu: 1) retriever najde chunky, 2) LLM vygeneruje odpověď
    chain = RetrievalQA.from_chain_type(
        llm=llm,                                        # jazykový model pro generování odpovědi
        chain_type="stuff",                             # "stuff" = všechny chunky vloží do jednoho promptu
        retriever=retriever,                            # hybridní retriever definovaný výše
        return_source_documents=True,                   # vrátí i zdrojové chunky (pro zobrazení stránek)
        chain_type_kwargs={"prompt": prompt_template},  # použije náš vlastní prompt
    )

    print("    Řetězec připraven.")
    return chain   # vrátí sestavený RAG řetězec připravený na dotazy


# ---------------------------------------------------------------------------
# KROK 5 — INTERAKTIVNÍ SMYČKA
# ---------------------------------------------------------------------------

def interactive_loop(chain):
    """
    Jednoduchá CLI smyčka — piš otázky, dostávej odpovědi.
    Napiš 'quit' nebo 'exit' pro ukončení.
    """
    print("\n" + "=" * 60)
    print("RAG chatbot připraven! Piš otázky (quit = konec).")
    print("=" * 60)

    while True:                                              # nekonečná smyčka dokud uživatel neukončí
        question = input("\nOtázka: ").strip()               # přečte vstup a odstraní mezery na krajích

        if not question:                                     # prázdný vstup — přeskočí
            continue
        if question.lower() in ("quit", "exit", "konec"):   # ukončovací slova
            print("Ukončuji…")
            break                                            # vyskočí ze smyčky

        result = chain.invoke({"query": question})           # pošle otázku do RAG řetězce a čeká na odpověď

        print(f"\nOdpověď:\n{result['result']}")             # zobrazí vygenerovanou odpověď

        sources = result.get("source_documents", [])         # vezme seznam zdrojových chunků
        if sources:                                          # pokud byly nějaké nalezeny
            pages = sorted({doc.metadata.get("page", "?") + 1 for doc in sources})  # unikátní čísla stránek (+1 protože PDF indexuje od 0)
            print(f"\nZdroje — stránky PDF: {pages}")        # zobrazí čísla stránek odkud odpověď pochází


# ---------------------------------------------------------------------------
# HLAVNÍ FUNKCE
# ---------------------------------------------------------------------------

def main(pdf_path: str = PDF_PATH):
    if not ANTHROPIC_API_KEY:                   # kontrola že klíč byl načten
        raise ValueError(
            "Nastav ANTHROPIC_API_KEY v .env souboru:\n"
            "  ANTHROPIC_API_KEY=sk-ant-..."
        )

    chroma_exists = Path(CHROMA_DIR).exists() and any(Path(CHROMA_DIR).iterdir())  # True pokud složka existuje a není prázdná

    # Chunky vždy načteme — BM25 je potřebuje při každém spuštění (neukládá se na disk)
    pages = load_pdf(pdf_path)          # KROK 1: načti PDF
    chunks = split_documents(pages)     # KROK 2: rozděl na chunky

    if chroma_exists:
        vectorstore = load_vectorstore(CHROMA_DIR)          # KROK 3a: načti existující vektory z disku
    else:
        vectorstore = build_vectorstore(chunks, CHROMA_DIR) # KROK 3b: vygeneruj a ulož vektory

    chain = build_rag_chain(vectorstore, chunks)  # KROK 4: sestav RAG řetězec
    interactive_loop(chain)                       # KROK 5: spusť interaktivní chatbot


if __name__ == "__main__":
    import sys                                                    # modul pro přístup k argumentům příkazové řádky
    pdf = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH         # pokud byl předán argument, použij ho; jinak výchozí cestu
    main(pdf)                                                     # spusť hlavní funkci
