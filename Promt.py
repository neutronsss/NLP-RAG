import re

def fungsi_input():
    pola_komponen = {
        "merk": r"\b(ASUS|Lenovo|Acer|HP|Dell|MSI|Apple|MacBook|Axioo|Advan|Gigabyte|Razer|Corsair|Kingston|Samsung|V-Color|Teamgroup|Adata|WD|Seagate|Intel|AMD|Nvidia)\b",
        "model": r"\b(TUF(?:\s*Gaming)?|Legion\s*\d*|ROG(?:\s*Strix|\s*Zephyrus|\s*Flow)?|Predator\s*\d*|Nitro\s*\d*|Pavilion|Ideapad|Victus\s*\d*|Omen\s*\d*|LOQ\s*\d*)\b",
        "cpu": r"\b(?:Intel\s+)?(?:Core\s+)?(?:[iI][3579]\s+gen\s+\d+|[iI][3579]\b|Core\s+Ultra\s+\d+|Ryzen\s+[3579]\s*\d*|Apple\s+M[1234](?:\s*(?:Pro|Max|Ultra))?)\b",
        "gpu": r"\b(?:Nvidia\s+)?(?:RTX\s*\d{4}(?:\s*Ti)?|GTX\s*\d{4}(?:\s*Ti)?|Radeon\s+RX\s*\d{4}\w*|Intel\s+Iris\s+X[ee])\b",
        "ram": r"\b\d+\s*GB\s*(?:RAM|DDR[45])?\b",
        "vram": r"\b(?:VRAM\s*\d+\s*GB|\d+\s*GB\s*VRAM)\b",
        "storage": r"\b\d+\s*(?:SSD|HDD|NVMe)\s*(?:GB|TB)\b",
        "harga": r"\b(?:harga|budget|rentang|di\s*bawah|di\s*atas|sekitar|harga\s*di)?\s*\d+(?:\s*(?:-|sampai|hingga)\s*\d+)?\s*(?:juta|jutaan|jt)\b",
    }

    terdeteksi = {
        "merk": [],
        "model": [],
        "spek": [],
        "harga": "-",
    }
    
    teks_sisa = input("Masukan Promt : ")

    for kategori, pola in pola_komponen.items():
        matches = re.findall(pola, teks_sisa, flags=re.IGNORECASE)
        for match in matches:
            val = match[0] if isinstance(match, tuple) else match
            val = val.strip()

            if kategori == "merk":
                if val not in terdeteksi["merk"]:
                    terdeteksi["merk"].append(val)
            elif kategori == "model":
                if val not in terdeteksi["model"]:
                    terdeteksi["model"].append(val)
            elif kategori == "harga":
                terdeteksi["harga"] = val
            else:
                if val not in terdeteksi["spek"]:
                    terdeteksi["spek"].append(val)

    # 1. Olah Nama Produk
    produk_list = terdeteksi["merk"] + terdeteksi["model"]
    produk = " ".join(produk_list) if produk_list else "Komponen / Produk Umum"

    spek_bersih = []
    for s in terdeteksi["spek"]:
        if not any(
            s in item and s != item for item in terdeteksi["spek"]
        ):
            spek_bersih.append(s)

    return {
        "Produk" : [produk],
        "Spesifikasi" : spek_bersih,
        "Harga" : [terdeteksi["harga"]]
    }
    
