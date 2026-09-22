import os
import sys
import shutil
import warnings
import logging

# Sembunyikan semua warning dan log eksternal agar tampilan terminal bersih
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("huggingface_hub").setLevel(logging.CRITICAL)
logging.getLogger("transformers").setLevel(logging.CRITICAL)
logging.getLogger("sentence_transformers").setLevel(logging.CRITICAL)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ==============================================================================
# 1. DUMMY DATA (Diperluas dari pricelist_ELS.pdf)
# Mencakup berbagai kategori: Vivobook, TUF Gaming, LOQ, IdeaPad, Macbook, Victus, dll.
# ==============================================================================
dummy_pricelist = [
    # ASUS
    (
        "ASUS Vivobook S S3407QA-IPSP152M (16/512GB) OHS Silver | "
        "Spesifikasi: Snapdragon X Plus X1 26 100 (2.97GHz / 8C / 8T), RAM 16GB LPDDR5X, SSD 512GB, "
        "Layar 14 inch 2.5K IPS 100% sRGB, Bobot 1.35kg tipis 1.59cm, OS Windows 11 + OHS 2024 + M365 Basic. "
        "Harga: Rp 12.999.000. Garansi 3 Tahun Internasional + APW."
    ),
    (
        "ASUS Vivobook Flip TP3407SA-OLED5152M (16/512GB) Mate Grey (OLED Touch 360°) | "
        "Spesifikasi: Intel Core Ultra 5 226V (4.5GHz / 8C / 8T), RAM 16GB LPDDR5, SSD 512GB, "
        "Layar 14 inch FHD OLED 100% DCI-P3 TouchScreen + Stylus, OS Windows 11 + OHS 2024. "
        "Harga: Rp 19.999.000. Garansi 3 Tahun Internasional + APW."
    ),
    (
        "ASUS TUF Gaming A15 FA506NCG-R735B1T-HM Graphite Black | "
        "Spesifikasi: AMD Ryzen 7 7445HS (4.7GHz / 6C / 12T), RAM 8GB DDR5, SSD 512GB, "
        "Layar 15.6 inch FHD IPS 144Hz, GPU NVIDIA GeForce RTX 3050 4GB, OS Windows 11 + OHS 2024. "
        "Harga: Rp 14.499.000. Garansi 2 Tahun."
    ),

    # APPLE
    (
        "Apple Macbook Neo 13 (8/256GB) Apple A18 pro | "
        "Spesifikasi: Apple A18 Pro, 8GB, 256GB, Layar 13 inch QHD, Apple Intelligence, "
        "Baterai hingga 16 jam. Pilihan warna: Silver, Indigo, Citrus, Blush. "
        "Harga: Rp 12.999.000. Garansi 1 Tahun."
    ),
    (
        "Apple Macbook Neo 13 (8/512GB) Apple A18 pro | "
        "Spesifikasi: Apple A18 Pro, 8GB, 512GB, Layar 13 inch QHD, Touch ID, Apple Intelligence, "
        "Baterai hingga 16 jam. Pilihan warna: Citrus, Indigo, Silver, Blush. "
        "Harga: Rp 14.999.000. Garansi 1 Tahun."
    ),

    # LENOVO
    (
        "Lenovo LOQ 15IAX9E-EHID Luna Grey (GAMING) | "
        "Spesifikasi: Intel Core i5-12450HX (4.4GHz / 8C / 12T), RAM 16GB DDR5, SSD 512GB (Up to 1TB), "
        "Layar 15.6 inch FHD IPS 100% sRGB 144Hz, GPU NVIDIA GeForce RTX 3050 6GB, OS Windows 11 + OHS 2024. "
        "Harga: Rp 16.999.000. Garansi 2 Tahun."
    ),
    (
        "Lenovo IdeaPad Slim 3 14IRU8 (8/512GB) OHS Abyss Blue / Artic Grey | "
        "Spesifikasi: Intel Core i3-1315U (4.5GHz / 6C / 8T), RAM 8GB LPDDR5, SSD 512GB, "
        "Layar 14 inch FHD, Keyboard Backlight, OS Windows 11 + OHS 2024 + M365 Basic. "
        "Harga: Rp 9.299.000. Garansi 2 Tahun."
    ),

    # ACER
    (
        "Acer GAMING Nitro V 15 ANV15-52-50AQ Black | "
        "Spesifikasi: Intel Core 5 210H (4.8GHz / 8C / 12T), RAM 16GB DDR5, SSD 512GB, "
        "Layar 15.6 inch FHD IPS 180Hz 100% sRGB, GPU NVIDIA GeForce RTX 3050 6GB, OS Windows 11 + OHS 2024. "
        "Harga: Rp 15.999.000. Garansi 2 Tahun."
    ),
    (
        "Acer Swift Air 14 AI SFA14-I31-57W2 Sage Green | "
        "Spesifikasi: Intel Core 5 320U (4.6GHz / 6C / 6T), RAM 8GB LPDDR5, SSD 512GB, "
        "Layar 14 inch WUXGA IPS 100% sRGB 120Hz, Bobot 1.19kg tipis 12.99mm, OS Windows 11 + OHS 2024. "
        "Harga: Rp 15.199.000. Garansi 2 Tahun."
    ),

    # HP & MSI & TABLET
    (
        "HP Victus GAMING 15-FA2717TX (16/512GB) Mica Silver | "
        "Spesifikasi: Intel Core i5-13420H (4.6GHz / 8C / 12T), RAM 16GB DDR4, SSD 512GB, "
        "Layar 15.6 inch FHD IPS 144Hz, GPU NVIDIA GeForce RTX 4050 6GB, OS Windows 11 + OHS 2024. "
        "Harga: Rp 17.999.000. Garansi 2 Tahun."
    ),
    (
        "MSI Modern 14-F13MG-498ID Grey (8/512GB) Slim Design | "
        "Spesifikasi: Intel Core i3-1315U (4.5GHz / 6C / 8T), RAM 8GB DDR4, SSD 512GB, "
        "Layar 14 inch FHD IPS, Bobot ringan, OS Windows 11. "
        "Harga: Rp 9.399.000. Garansi 2 Tahun."
    ),
    (
        "Huawei MatePad 11.5 (8/256GB) Space Grey / Violet | "
        "Spesifikasi: Tablet Kirin T82B, RAM 8GB, Storage 256GB, Layar 11.5 inch 2.5K 120Hz, "
        "Baterai 10.100 mAh, Kamera Belakang 13MP, Kamera Depan 8MP, Bundle Keyboard + M-Pencil. "
        "Harga: Rp 8.999.000. Garansi 1 Tahun."
    ),
    (
        "Polytron Luxia (8/256GB/i3) Obsidian Grey | "
        "Spesifikasi: Intel Core i3-1215U (4.4GHz / 6C / 8T), RAM 8GB DDR4, SSD 256GB, "
        "Layar 14 inch WUXGA (1920x1200) IPS, OS Windows 11, Bobot 1.4kg. "
        "Harga: Rp 6.499.000. Garansi 2 Tahun."
    )
]

def create_documents(raw_texts):
    """Mengubah list of strings menjadi list Document LangChain"""
    return [Document(page_content=text, metadata={"id": idx}) for idx, text in enumerate(raw_texts)]

def chunk_documents(documents):
    """
    TUGAS ORANG 2 - Bagian 1: Chunking Teks
    Menyiapkan modul LangChain untuk memecah deskripsi produk ke format dokumen siap pakai.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        separators=[" | ", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def build_vector_store(chunks, persist_directory="./chroma_db", reset=True):
    """
    TUGAS ORANG 2 - Bagian 2 & 3: Embedding Model & ChromaDB
    - Model embedding: HuggingFace all-MiniLM-L6-v2
    - Inisialisasi ChromaDB lokal & menyimpan vektor
    """
    if reset and os.path.exists(persist_directory):
        try:
            shutil.rmtree(persist_directory)
        except Exception:
            pass

    print(f"[*] Menginisialisasi model embedding 'all-MiniLM-L6-v2'...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print(f"[*] Menyimpan {len(chunks)} chunks ke ChromaDB lokal di: '{persist_directory}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore

def setup_retriever(vectorstore, k=2):
    """
    TUGAS ORANG 2 - Bagian 3: Membangun fungsi similarity search (retriever)
    """
    return vectorstore.as_retriever(search_kwargs={"k": k})

def test_retrieval(retriever, query):
    """Fungsi pembantu untuk menguji dan menampilkan hasil retrieval"""
    print("\n" + "=" * 70)
    print(f"Query Pengguna: \"{query}\"")
    print("-" * 70)
    
    results = retriever.invoke(query)
    
    print(f"Hasil Retrieval ({len(results)} dokumen teratas):")
    for i, doc in enumerate(results, 1):
        print(f"\n[Dokumen {i}]:")
        print(doc.page_content)
    print("=" * 70)
    return results

if __name__ == "__main__":
    print("==========================================================")
    print("      PIPELINE VECTOR DATABASE & RETRIEVAL (ORANG 2)      ")
    print("==========================================================")
    
    # 1. Dokumen awal
    docs = create_documents(dummy_pricelist)
    print(f"[+] Total data produk: {len(docs)} item.")
    
    # 2. Chunking
    chunks = chunk_documents(docs)
    print(f"[+] Total chunks terbentuk: {len(chunks)} chunks.")
    
    # 3. Vector Database
    vector_db = build_vector_store(chunks, persist_directory="./chroma_db", reset=True)
    
    # 4. Inisialisasi Retriever
    retriever = setup_retriever(vector_db, k=2)
    print("[+] Retriever siap menerima query pengguna!\n")
    
    # 5. Interactive Query Loop
    print("-" * 60)
    print("Ketik query pencarian Anda di bawah (contoh: 'asus vivobook').")
    print("Ketik 'exit' atau 'q' untuk keluar.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nMasukkan query Anda: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "q", "quit", "keluar"]:
                print("Terima kasih, program selesai.")
                break
            test_retrieval(retriever, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nProgram dihentikan.")
            break
