import json
import re
import subprocess
import urllib.parse
import urllib.request
import zipfile
from typing import NamedTuple

subprocess.run(['pip', 'install', '-q', 'mineru[full]==4.0.8'], check=True)
subprocess.run(['mineru-kit', 'models', 'download', '--tier', 'standard'], check=True)

COURSE_DISK = 'https://disk.yandex.ru/d/BxazXSphID0jFw'
DISK_DOWNLOAD_API = 'https://cloud-api.yandex.net/v1/disk/public/resources/download'
BIBLIOGRAPHY_HEADING = re.compile(r'^#+\s*Литература\s*$', re.M)
CHAPTER_HEADING = re.compile(r'^#+\s*Глава\s*\d+\s+[^.\s]', re.M)


class Book(NamedTuple):
    disk_file: str
    pages: str


BOOKS = {
    'efron': Book('efron_li_metallovedenie_v_bolshoi_metallurgii_trubnye_stali (1).pdf', '96-285'),
    'odesskii': Book('odesskii_pd_smirnov_la_kulik_dv_mikrolegirovannye_stali_dlia.pdf', '31-125'),
}


def without_bibliography(text):
    for heading in reversed(list(BIBLIOGRAPHY_HEADING.finditer(text))):
        chapter = CHAPTER_HEADING.search(text, heading.end())
        end = chapter.start() if chapter else len(text)
        print(f'bibliography removed: {end - heading.start()} characters')
        text = text[:heading.start()] + text[end:]
    return text


def download(disk_file, path):
    query = urllib.parse.urlencode({'public_key': COURSE_DISK, 'path': '/' + disk_file})
    urllib.request.urlretrieve(json.load(urllib.request.urlopen(f'{DISK_DOWNLOAD_API}?{query}'))['href'], path)


for name, book in BOOKS.items():
    pdf = f'/tmp/{name}.pdf'
    download(book.disk_file, pdf)
    subprocess.run(['mineru-kit', 'parse', pdf, '-o', f'/tmp/{name}.zip', '--pages', book.pages,
                    '--ocr-mode', 'ocr', '--format', 'zip'], check=True)
    with zipfile.ZipFile(f'/tmp/{name}.zip') as archive:
        markdown = next(member for member in archive.namelist() if member.endswith('.md'))
        text = without_bibliography(archive.read(markdown).decode('utf-8'))
        with open(f'/kaggle/working/{name}.txt', 'w', encoding='utf-8') as output:
            output.write(text)
