import requests
import tarfile
import os
import math

#List of blacklist archive URLs
BLACKLIST_ARCHIVES = [
    "https://dsi.ut-capitole.fr/blacklists/download/ads.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/adult.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/aggressive.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/agressif.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/arjel.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/associations_religieuses.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/astrology.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/audio-video.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/bank.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/bitcoin.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/blacklists.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/blacklists_for_dansguardian.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/blacklists_for_fortigate.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/blacklists_for_pfsense.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/blacklists_for_pfsense_reducted.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/blog.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/catalogue-biu-toulouse.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/celebrity.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/chat.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/child.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/cleaning.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/cooking.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/cryptojacking.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/dangerous_material.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/dating.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/ddos.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/dialer.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/doh.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/domains.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/download.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/drogue.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/drugs.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/dynamic-dns.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/educational_games.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/examen_pix.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/fakenews.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/filehosting.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/financial.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/forums.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/gambling.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/games.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/hacking.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/indisponible.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/jobsearch.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/jstor.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/lingerie.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/liste_blanche.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/liste_bu.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/local.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/malware.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/manga.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/marketingware.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/mixed_adult.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/mobile-phone.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/phishing.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/porn.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/press.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/proxy.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/publicite.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/radio.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/reaffected.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/redirector.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/remote-control.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/residential-proxies.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/sect.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/sexual_education.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/shopping.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/shortener.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/social_networks.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/special.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/sports.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/stalkerware.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/strict_redirector.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/strong_redirector.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/translation.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/tricheur.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/tricheur_pix.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/update.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/verisign.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/violence.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/vpn.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/warez.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/webhosting.tar.gz",
    "https://dsi.ut-capitole.fr/blacklists/download/webmail.tar.gz",
]

BASE_TMP_DIR = "./archive"
OUTPUT_FILE_PREFIX = "toulouse_blacklist_part"
MAX_DOMAINS_PER_FILE = 100000  # Change as needed (e.g., 100000, 500000, etc)

def download_and_extract(url, base_tmp_dir):
    archive_name = os.path.basename(url)
    archive_tmp_dir = os.path.join(base_tmp_dir, archive_name.replace('.tar.gz', ''))
    os.makedirs(archive_tmp_dir, exist_ok=True)
    local_tar = os.path.join(archive_tmp_dir, archive_name)
    print(f"Downloading {url} ...")
    try:
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            with open(local_tar, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Saved archive to {local_tar}")
            with tarfile.open(local_tar, "r:gz") as tar:
                tar.extractall(path=archive_tmp_dir)
            print(f"Extracted {local_tar} to {archive_tmp_dir}")
        else:
            print(f"Failed to download {url}, status code: {r.status_code}")
    except Exception as e:
        print(f"Error downloading or extracting {url}: {e}")
    return archive_tmp_dir

def collect_domains(all_tmp_dirs):
    domain_set = set()
    for extract_dir in all_tmp_dirs:
        for root, dirs, files in os.walk(extract_dir):
            if "domains" in files:
                file_path = os.path.join(root, "domains")
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            domain = line.strip()
                            if (domain and not domain.startswith("#") and
                                "." in domain and " " not in domain and len(domain) > 3):
                                domain_set.add(domain)
                    print(f"Added domains from {file_path}")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
    return sorted(domain_set)

def save_domains_split(domain_list, file_prefix, max_per_file):
    total = len(domain_list)
    num_files = math.ceil(total / max_per_file)
    for i in range(num_files):
        start = i * max_per_file
        end = start + max_per_file
        out_file = f"{file_prefix}{i+1}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for domain in domain_list[start:end]:
                f.write(domain + "\n")
        print(f"Wrote {end-start} domains to {out_file}")

def main():
    os.makedirs(BASE_TMP_DIR, exist_ok=True)
    all_tmp_dirs = []
    for url in BLACKLIST_ARCHIVES:
        tmp_dir = download_and_extract(url, BASE_TMP_DIR)
        all_tmp_dirs.append(tmp_dir)
    domains = collect_domains(all_tmp_dirs)
    save_domains_split(domains, OUTPUT_FILE_PREFIX, MAX_DOMAINS_PER_FILE)
    print("Done. Each TXT file is ready for Pi-hole gravity import or IPFire URL FILTER.")

if __name__ == "__main__":
    main()
