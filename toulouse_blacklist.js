import fetch from 'node-fetch';
import tar from 'tar';
import fs from 'fs';
import path from 'path';

//List of blacklist archive URLs
const BLACKLIST_ARCHIVES = [
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
];

const BASE_TMP_DIR = "./archive";
const OUTPUT_FILE_PREFIX = "toulouse_blacklist_part";
const MAX_DOMAINS_PER_FILE = 100000; // Adjust as needed

async function downloadAndExtract(url, baseTmpDir) {
    const archiveName = path.basename(url);
    const archiveTmpDir = path.join(baseTmpDir, archiveName.replace('.tar.gz', ''));
    fs.mkdirSync(archiveTmpDir, { recursive: true });
    const localTar = path.join(archiveTmpDir, archiveName);
    console.log(`Downloading ${url} ...`);
    try {
        const res = await fetch(url);
        if (res.ok) {
            const fileStream = fs.createWriteStream(localTar);
            await new Promise((resolve, reject) => {
                res.body.pipe(fileStream);
                res.body.on("error", reject);
                fileStream.on("finish", resolve);
            });
            console.log(`Saved archive to ${localTar}`);
            await tar.x({ file: localTar, cwd: archiveTmpDir });
            console.log(`Extracted ${localTar} to ${archiveTmpDir}`);
        } else {
            console.log(`Failed to download ${url}, status code: ${res.status}`);
        }
    } catch (e) {
        console.log(`Error downloading or extracting ${url}: ${e}`);
    }
    return archiveTmpDir;
}

function collectDomains(allTmpDirs) {
    const domainSet = new Set();
    for (const extractDir of allTmpDirs) {
        function walkSync(dir) {
            fs.readdirSync(dir, { withFileTypes: true }).forEach((dirent) => {
                const filePath = path.join(dir, dirent.name);
                if (dirent.isDirectory()) {
                    walkSync(filePath);
                } else if (dirent.name === "domains") {
                    try {
                        const data = fs.readFileSync(filePath, "utf-8");
                        data.split('\n').forEach(line => {
                            const domain = line.trim();
                            if (
                                domain &&
                                !domain.startsWith("#") &&
                                domain.includes(".") &&
                                !domain.includes(" ") &&
                                domain.length > 3
                            ) {
                                domainSet.add(domain);
                            }
                        });
                        console.log(`Added domains from ${filePath}`);
                    } catch (e) {
                        console.log(`Error reading ${filePath}: ${e}`);
                    }
                }
            });
        }
        walkSync(extractDir);
    }
    return Array.from(domainSet).sort();
}

function saveDomainsSplit(domainList, filePrefix, maxPerFile) {
    const total = domainList.length;
    const numFiles = Math.ceil(total / maxPerFile);
    for (let i = 0; i < numFiles; i++) {
        const start = i * maxPerFile;
        const end = Math.min(start + maxPerFile, total);
        const outFile = `${filePrefix}${i + 1}.txt`;
        fs.writeFileSync(outFile, domainList.slice(start, end).join('\n') + '\n', "utf-8");
        console.log(`Wrote ${end - start} domains to ${outFile}`);
    }
}

async function main() {
    fs.mkdirSync(BASE_TMP_DIR, { recursive: true });
    const allTmpDirs = [];
    for (const url of BLACKLIST_ARCHIVES) {
        const tmpDir = await downloadAndExtract(url, BASE_TMP_DIR);
        allTmpDirs.push(tmpDir);
    }
    const domains = collectDomains(allTmpDirs);
    saveDomainsSplit(domains, OUTPUT_FILE_PREFIX, MAX_DOMAINS_PER_FILE);
    console.log("Done. Each TXT file is ready for Pi-hole gravity import or IPFire URL FILTER.");
}

main();
