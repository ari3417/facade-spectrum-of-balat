!pip install pandas geopandas osmnx shapely scikit-learn beautifulsoup4 lxml
import os
import pandas as pd
import geopandas as gpd
import osmnx as ox
import json
import cv2
import numpy as np
import zipfile
import math
import warnings
import re
import base64
import random
from shapely.geometry import Point, mapping
from sklearn.cluster import KMeans
from bs4 import BeautifulSoup

# ==========================================
# 1. AYARLAR VE KLASÖR YOLLARI
# ==========================================
EXCEL_V4 = 'Balat_Soyut_Tipoloji_Analizi(5).xlsx'
EXCEL_GIS = 'Balat_Kesinlesmis_Bina_Koordinatlari.xlsx'
KLASOR_SOYUT_KIRPILMIS = '/content/drive/MyDrive/balat-soyut-kirpilmis/'
KLASOR_SEMBOLIK_ABSTRACT = '/content/drive/MyDrive/balat_sembolik_binalar_abstract/'

# Colab Cache problemini aşmak için versiyonlu isimler:
CIKTI_HTML = 'Balat_Interactive_Dashboard_v7.html'
ZIP_ISMI = 'Balat_Dashboard_v7.zip'

WEB_IMAGES_DIR = "web_images"
os.makedirs(WEB_IMAGES_DIR, exist_ok=True)

warnings.filterwarnings('ignore', category=UserWarning)

# ==========================================
# 2. VERİTABANI BİRLEŞTİRME VE TEMİZLEME
# ==========================================
print("[1/5] Veritabanları ID numaraları üzerinden birleştiriliyor...")
try:
    df_v4 = pd.read_excel(EXCEL_V4)
    df_gis = pd.read_excel(EXCEL_GIS)
except Exception as e:
    print("HATA: Excel dosyaları okunamadı. Yolları kontrol edin.", e)

def id_cikar(isim):
    try: return int(str(isim).split('_')[1])
    except: return 0

df_v4['Cikartilan_ID'] = df_v4['Fotoğraf İsmi'].apply(id_cikar)
df_gis['Bina_ID'] = pd.to_numeric(df_gis['Bina_ID'], errors='coerce')
df = pd.merge(df_gis, df_v4, left_on='Bina_ID', right_on='Cikartilan_ID', how='inner')
df = df.dropna(subset=['Bina_Lat', 'Bina_Lon'])

# ==========================================
# 3. YOLA PARALEL ÇATI POLİGONLARI VE YOL KATMANI
# ==========================================
print("[2/5] Binalar ve yollar OpenStreetMap'ten çekiliyor...")
place_name = "Balat, Istanbul, Turkey"
binalar_gdf = ox.features_from_place(place_name, tags={"building": True})
binalar_gdf = binalar_gdf.to_crs(epsg=32635)

yollar_gdf = ox.features_from_place(place_name, tags={"highway": True})
yollar_gdf = yollar_gdf[yollar_gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
yollar_gdf = yollar_gdf.to_crs(epsg=4326)
yollar_geojson = yollar_gdf.to_json()

geometry = [Point(xy) for xy in zip(df['Bina_Lon'], df['Bina_Lat'])]
noktalar_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326").to_crs(epsg=32635)

binalar_gdf = binalar_gdf.reset_index()

for coord, group in noktalar_gdf.groupby(['Bina_Lat', 'Bina_Lon']):
    pt = group.geometry.iloc[0]
    circle = pt.buffer(25)
    intersecting = binalar_gdf[binalar_gdf.intersects(circle)]

    if len(intersecting) == 0:
        nearest_idx = binalar_gdf.geometry.distance(pt).idxmin()
        bldg_indices = [nearest_idx]
    else:
        intersecting = intersecting.copy()
        intersecting['dist'] = intersecting.geometry.distance(pt)
        intersecting = intersecting.sort_values('dist')
        bldg_indices = intersecting.index.tolist()

    for i, (idx, row) in enumerate(group.iterrows()):
        assigned_bldg = bldg_indices[i % len(bldg_indices)]
        noktalar_gdf.loc[idx, 'index_right'] = assigned_bldg

eslesen_veriler = noktalar_gdf.copy()
eslesen_veriler = eslesen_veriler.to_crs(epsg=4326)
binalar_gdf = binalar_gdf.to_crs(epsg=4326)
eslesen_veriler['poly_geom'] = binalar_gdf.loc[eslesen_veriler['index_right'], 'geometry'].values

# ==========================================
# 4. WEBP FORMATINDA DIŞA AKTARMA
# ==========================================
print("[3/5] Fotoğraflar yüksek çözünürlüklü WebP formatında klasöre çıkarılıyor...")

def get_hex_and_save_webp(image_path, output_filename):
    if not os.path.exists(image_path): return "#cccccc", ""

    img = None
    try:
        with open(image_path, "rb") as f: chunk = f.read()
        chunk_arr = np.frombuffer(chunk, dtype=np.uint8)
        img = cv2.imdecode(chunk_arr, cv2.IMREAD_UNCHANGED)
    except: pass

    if img is None: return "#cccccc", ""

    if len(img.shape) == 3 and img.shape[2] == 4:
        vis = img[img[:, :, 3] > 0]
        avg = np.average(vis, axis=0) if len(vis) > 0 else [200, 200, 200]
    else:
        h, w = img.shape[:2]
        avg = np.average(np.average(img[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)], axis=0), axis=0)
    hex_color = "#{:02x}{:02x}{:02x}".format(int(avg[2]), int(avg[1]), int(avg[0]))

    h, w = img.shape[:2]
    if h > 400:
        oran = 400.0 / h
        yeni_w = max(int(w * oran), 8)
        img_kucuk = cv2.resize(img, (yeni_w, 400), interpolation=cv2.INTER_AREA)
    else:
        img_kucuk = img

    output_path = os.path.join(WEB_IMAGES_DIR, output_filename)
    try:
        _, buf = cv2.imencode('.webp', img_kucuk, [int(cv2.IMWRITE_WEBP_QUALITY), 85])
        with open(output_path, "wb") as f: f.write(buf)
    except: pass

    return hex_color, f"{WEB_IMAGES_DIR}/{output_filename}"

def hex_to_rgb(hx): return [int(hx.lstrip('#')[0:2],16), int(hx.lstrip('#')[2:4],16), int(hx.lstrip('#')[4:6],16)]
def rgb_to_hex(rgb): return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

rgb_list = []
ozellikler = []
hacimler = []
gecerli_indices = []

for idx, row in eslesen_veriler.iterrows():
    dosya_yolu = os.path.join(KLASOR_SOYUT_KIRPILMIS, str(row['Dosya_Adi']))
    bina_id = str(row['Bina_ID'])
    output_filename = f"bina_{bina_id}.webp"

    hex_renk, img_link = get_hex_and_save_webp(dosya_yolu, output_filename)
    if not img_link: continue

    try: aks = sum([int(s) for s in re.findall(r'\d+', str(row['Duvar / Cumba Aksı']))])
    except: aks = 2
    hacim = max(1, int(row['Kat Sayısı']) * aks)
    hacimler.append(hacim)

    malzemeler = str(row['Kat Malzemeleri']).split(' | ')
    foto_ismi = str(row.get('Fotoğraf İsmi', ''))
    sokak_ismi = "Bilinmiyor"
    for part in foto_ismi.split('_'):
        if "Cd" in part or "Sk" in part or "Sok" in part or "Cad" in part:
            sokak_ismi = part; break
    if sokak_ismi == "Bilinmiyor":
        parts = foto_ismi.split('_')
        if len(parts) > 2: sokak_ismi = parts[2]

    kat_val = str(row.get('Kat Sayısı', 'Belirtilmemiş'))
    if str(kat_val).strip().lower() == 'nan': kat_val = 'Belirtilmemiş'
    else: kat_val = str(kat_val).replace('.0', '')

    zemin_val = 'Belirtilmemiş'
    for olasi_kolon in ['Zemin Kat Fonksiyonu', 'Zemin Kat (Fonksiyon)', 'Zemin Kat', 'Fonksiyon']:
        if olasi_kolon in row.index and pd.notna(row[olasi_kolon]) and str(row[olasi_kolon]).strip().lower() != 'nan':
            zemin_val = str(row[olasi_kolon]).strip(); break

    ozellikler.append({
        "id": bina_id, "img": img_link, "malzeme": malzemeler[0] if malzemeler else "Sıvalı",
        "renk_orj": hex_renk, "sokak": sokak_ismi, "kat": kat_val, "zemin": zemin_val
    })
    rgb_list.append(hex_to_rgb(hex_renk))
    gecerli_indices.append(idx)

eslesen_veriler = eslesen_veriler.loc[gecerli_indices].reset_index(drop=True)

X = np.array(rgb_list)
W = np.array(hacimler)
palet_10 = KMeans(n_clusters=min(10, len(X)), random_state=42).fit(X, sample_weight=W)
palet_5 = KMeans(n_clusters=min(5, len(X)), random_state=42).fit(X, sample_weight=W)
palet_3 = KMeans(n_clusters=min(3, len(X)), random_state=42).fit(X, sample_weight=W)
palet_1 = KMeans(n_clusters=1, random_state=42).fit(X, sample_weight=W)

unique_rgb = np.unique(X, axis=0)
palet_25_model = KMeans(n_clusters=min(25, len(unique_rgb)), random_state=42).fit(unique_rgb)
labels_unique = palet_25_model.labels_

def get_vibrancy(c):
    r, g, b = float(c[0]), float(c[1]), float(c[2])
    return (max(r, g, b) * 3.0) + (r * 1.5 + g * 1.0 - b * 0.5)

palet_25_colors = []
for i in range(palet_25_model.n_clusters):
    cluster_colors = unique_rgb[labels_unique == i]
    best_c = palet_25_model.cluster_centers_[i] if len(cluster_colors) == 0 else max(cluster_colors, key=get_vibrancy)
    palet_25_colors.append(rgb_to_hex(best_c))

labels_25_full = palet_25_model.predict(X)
dominant_cluster_idx = np.argmax(np.bincount(labels_25_full))
most_dominant_color = palet_25_colors[dominant_cluster_idx]

# ==========================================
# 5. KML DOSYASINDAN MANUEL PİNLERİ YÜKLEME VE FAILSAFE EŞLEŞTİRME
# ==========================================
print("KML dosyasından sembolik binalar eşleştiriliyor (Zorunlu Failsafe devrede)...")

try: mevcut_dosyalar = os.listdir(KLASOR_SEMBOLIK_ABSTRACT)
except: mevcut_dosyalar = []

def normalize(s):
    s = str(s).lower()
    for old, new in [('ş','s'), ('ğ','g'), ('ı','i'), ('i','i'), ('ö','o'), ('ç','c'), ('ü','u'), (' ', '_'), ('-', '_')]:
        s = s.replace(old, new)
    return s

def translate_to_english(tr_name):
    name = str(tr_name).replace('_', ' ').lower()
    dict_map = {
        'fener rum lisesi': 'Fener Greek High School',
        'ahrida sinagogu': 'Ahrida Synagogue',
        'dimitrie cantemir müzesi': 'Dimitrie Cantemir Museum',
        'dimitrie cantemir muzesi': 'Dimitrie Cantemir Museum',
        'fethiye camii': 'Fethiye Mosque',
        'camhane': 'Camhane',
        'fener rum patrikhanesi': 'Fener Greek Patriarchate',
        'tevkii cafer camii': 'Tevkii Cafer Mosque',
        'demir kilise': 'Bulgarian Iron Church',
        'yavuz selim camii': 'Yavuz Selim Mosque',
        'papazın evi': 'Priest\'s House',
        'papazin evi': 'Priest\'s House',
        'meryem ana kilisesi': 'Virgin Mary Church',
        'surp hreşdagabet kilisesi': 'Surp Hresdagabet Church',
        'surp hresdagabet kilisesi': 'Surp Hresdagabet Church',
        'ismailağa camii': 'Ismailaga Mosque',
        'ismailaga camii': 'Ismailaga Mosque',
        'ismail aga camii': 'Ismailaga Mosque',
    }
    if name in dict_map: return dict_map[name]

    name = name.replace('kilisesi', 'church').replace('kilise', 'church')
    name = name.replace('camii', 'mosque').replace('cami', 'mosque')
    name = name.replace('sinagogu', 'synagogue').replace('sinagog', 'synagogue')
    name = name.replace('müzesi', 'museum').replace('muzesi', 'museum')
    name = name.replace('patrikhanesi', 'patriarchate').replace('lisesi', 'high school')
    name = name.replace('evi', 'house')
    return " ".join([w.capitalize() for w in name.split()])

manuel_pin_data = []
manuel_poly_indices = set()
eslesen_dosyalar_kumesi = set()

if os.path.exists("binalar.kml"):
    try:
        with open("binalar.kml", "r", encoding="utf-8") as file:
            soup = BeautifulSoup(file, "xml")
            for pm in soup.find_all("Placemark"):
                name_tag, coord_tag = pm.find("name"), pm.find("coordinates")
                if name_tag and coord_tag:
                    isim = name_tag.text.strip()
                    coords = coord_tag.text.strip().split(',')
                    if len(coords) >= 2:
                        lon, lat = float(coords[0]), float(coords[1])
                        pin_pt = Point(lon, lat)

                        dist_s = binalar_gdf.geometry.distance(pin_pt)
                        if dist_s.min() < 0.0003: manuel_poly_indices.add(dist_s.idxmin())

                        hedef = normalize(isim)
                        eslesen_dosya = None
                        d_name_for_trans = isim

                        # 1. Tam Eşleşme
                        for d in mevcut_dosyalar:
                            if d.lower().endswith(('.png', '.jpg', '.jpeg')):
                                d_name = os.path.splitext(d)[0]
                                if normalize(d_name) == hedef:
                                    eslesen_dosya = d
                                    d_name_for_trans = d_name
                                    break

                        # 2. Kısmi Eşleşme (Kelime Kesişimi)
                        if not eslesen_dosya:
                            hedef_kelimeler = set(hedef.split('_'))
                            en_yuksek_skor = 0
                            for d in mevcut_dosyalar:
                                if d.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    d_name = os.path.splitext(d)[0]
                                    d_kelimeler = set(normalize(d_name).split('_'))
                                    ortak = hedef_kelimeler.intersection(d_kelimeler)
                                    if len(ortak) > en_yuksek_skor and len(ortak) >= 1:
                                        en_yuksek_skor = len(ortak)
                                        eslesen_dosya = d
                                        d_name_for_trans = d_name

                        # 3. Agresif Karakter Eşleşmesi (Boşluksuz İçinde Geçme)
                        if not eslesen_dosya:
                            hedef_str = hedef.replace('_', '')
                            for d in mevcut_dosyalar:
                                if d.lower().endswith(('.png', '.jpg', '.jpeg')):
                                    d_name = os.path.splitext(d)[0]
                                    d_str = normalize(d_name).replace('_', '')
                                    if hedef_str in d_str or d_str in hedef_str:
                                        eslesen_dosya = d
                                        d_name_for_trans = d_name
                                        break

                        if eslesen_dosya:
                            eslesen_dosyalar_kumesi.add(eslesen_dosya)
                            yol = os.path.join(KLASOR_SEMBOLIK_ABSTRACT, eslesen_dosya)
                            img = None
                            try:
                                with open(yol, "rb") as f: chunk = f.read()
                                img = cv2.imdecode(np.frombuffer(chunk, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                            except: pass

                            if img is not None:
                                h, w = img.shape[:2]
                                if h > 400: img_kucuk = cv2.resize(img, (max(int(w*(400.0/h)), 8), 400), interpolation=cv2.INTER_AREA)
                                else: img_kucuk = img

                                hex_c = "#888888"
                                if len(img.shape) == 3 and img.shape[2] == 4:
                                    vis = img[img[:, :, 3] > 0]
                                    if len(vis) > 0:
                                        avg = np.average(vis, axis=0)
                                        hex_c = "#{:02x}{:02x}{:02x}".format(int(avg[2]), int(avg[1]), int(avg[0]))
                                else:
                                    hh, ww = img.shape[:2]
                                    avg = np.average(np.average(img[int(hh*0.4):int(hh*0.6), int(ww*0.4):int(ww*0.6)], axis=0), axis=0)
                                    hex_c = "#{:02x}{:02x}{:02x}".format(int(avg[2]), int(avg[1]), int(avg[0]))

                                _, buf = cv2.imencode('.webp', img_kucuk, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
                                b64_str = base64.b64encode(buf).decode('utf-8')
                                bulunan_link = f"data:image/webp;base64,{b64_str}"

                                en_isim = translate_to_english(d_name_for_trans)
                                manuel_pin_data.append({"isim": isim, "en_isim": en_isim, "lat": lat, "lon": lon, "b64": bulunan_link, "renk_orj": hex_c})
    except Exception as e: print(f"KML okuma hatası: {e}")

# KML'DE BULUNMAYAN AMA KLASÖRDE OLAN DOSYALARI ZORLA EKLEME (FAILSAFE)
kalan_dosyalar = [d for d in mevcut_dosyalar if d.lower().endswith(('.png', '.jpg', '.jpeg')) and d not in eslesen_dosyalar_kumesi]

for d in kalan_dosyalar:
    d_name = os.path.splitext(d)[0]
    yol = os.path.join(KLASOR_SEMBOLIK_ABSTRACT, d)
    img = None
    try:
        with open(yol, "rb") as f: chunk = f.read()
        img = cv2.imdecode(np.frombuffer(chunk, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except: pass

    if img is not None:
        h, w = img.shape[:2]
        if h > 400: img_kucuk = cv2.resize(img, (max(int(w*(400.0/h)), 8), 400), interpolation=cv2.INTER_AREA)
        else: img_kucuk = img

        hex_c = "#888888"
        if len(img.shape) == 3 and img.shape[2] == 4:
            vis = img[img[:, :, 3] > 0]
            if len(vis) > 0:
                avg = np.average(vis, axis=0)
                hex_c = "#{:02x}{:02x}{:02x}".format(int(avg[2]), int(avg[1]), int(avg[0]))
        else:
            hh, ww = img.shape[:2]
            avg = np.average(np.average(img[int(hh*0.4):int(hh*0.6), int(ww*0.4):int(ww*0.6)], axis=0), axis=0)
            hex_c = "#{:02x}{:02x}{:02x}".format(int(avg[2]), int(avg[1]), int(avg[0]))

        _, buf = cv2.imencode('.webp', img_kucuk, [int(cv2.IMWRITE_WEBP_QUALITY), 80])
        b64_str = base64.b64encode(buf).decode('utf-8')
        bulunan_link = f"data:image/webp;base64,{b64_str}"

        en_isim = translate_to_english(d_name)
        # Eğer KML'de unutulmuşsa, Balat merkezine (rastgele hafif kaydırarak) yerleştiriyoruz
        # Böylece fotoğraflar kesinlikle dahil olup kolajda eksiksiz çıkacak!
        rlat = 41.0315 + (random.random() - 0.5) * 0.002
        rlon = 28.9480 + (random.random() - 0.5) * 0.002
        manuel_pin_data.append({"isim": d_name, "en_isim": en_isim, "lat": rlat, "lon": rlon, "b64": bulunan_link, "renk_orj": hex_c})

manuel_pin_json = json.dumps(manuel_pin_data)

# ==========================================
# 6. JSON FEATURE'LARI OLUŞTURMA VE ÇAKIŞANLARI GİZLEME
# ==========================================
features_geojson = []
poly_groups = {}
for i, row in enumerate(eslesen_veriler.itertuples()):
    if row.index_right not in poly_groups: poly_groups[row.index_right] = []
    poly_groups[row.index_right].append(i)

for poly_idx, indices in poly_groups.items():
    if poly_idx in manuel_poly_indices: continue

    geom = binalar_gdf.loc[poly_idx, 'geometry']
    coords = mapping(geom)
    n = len(indices)
    step_size = 0.00004
    for j, i in enumerate(indices):
        factor = j - (n - 1) / 2.0
        lat = geom.centroid.y + math.sin(0) * step_size * factor
        lon = geom.centroid.x + math.cos(0) * step_size * factor

        props = ozellikler[i].copy()
        props["c10"] = rgb_to_hex(palet_10.cluster_centers_[palet_10.labels_[i]])
        props["c5"] = rgb_to_hex(palet_5.cluster_centers_[palet_5.labels_[i]])
        props["c3"] = rgb_to_hex(palet_3.cluster_centers_[palet_3.labels_[i]])
        props["c1"] = rgb_to_hex(palet_1.cluster_centers_[palet_1.labels_[i]])
        props["c25_closest"] = palet_25_colors[labels_25_full[i]]
        props["center_lat"] = lat
        props["center_lon"] = lon

        features_geojson.append({"type": "Feature", "geometry": coords, "properties": props})

def sort_vibrancy_feature(f):
    c = f['properties']['c25_closest']
    freq = color_counts_dict[c]
    hx = f['properties']['renk_orj']
    r, g, b = int(hx.lstrip('#')[0:2],16), int(hx.lstrip('#')[2:4],16), int(hx.lstrip('#')[4:6],16)
    rf, gf, bf = r/255.0, g/255.0, b/255.0
    cmax, cmin = max(rf, gf, bf), min(rf, gf, bf)
    delta = cmax - cmin
    if delta == 0: h = 0
    elif cmax == rf: h = ((gf - bf) / delta) % 6
    elif cmax == gf: h = (bf - rf) / delta + 2
    else: h = (rf - gf) / delta + 4
    h = round(h * 60)
    if h < 0: h += 360
    vibrancy = (max(r, g, b) - min(r, g, b)) * 3.0 + r * 1.5 + g * 1.0 - b * 0.5
    return (-freq, h, -vibrancy)

color_counts_dict = {}
for f in features_geojson:
    c = f['properties']['c25_closest']
    color_counts_dict[c] = color_counts_dict.get(c, 0) + 1

features_geojson.sort(key=sort_vibrancy_feature)
geojson_data = json.dumps({"type": "FeatureCollection", "features": features_geojson})
palet_json = json.dumps(palet_25_colors)

# ==========================================
# 7. BALAT SINIRI (MASKE)
# ==========================================
print("[4/5] Balat resmi sınırları OSM'den çekiliyor...")
sinir_gdf = ox.geocode_to_gdf("Balat, Istanbul, Turkey")
balat_geojson = sinir_gdf.to_json()
min_lon, min_lat, max_lon, max_lat = binalar_gdf.total_bounds

# ==========================================
# 8. HTML, CSS, JS OLUŞTURMA
# ==========================================
print("[5/5] İnteraktif Kentsel Arayüz (Dashboard) kodlanıyor...")

html_icerik = f"""
<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Facade Spectrum of Balat</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Fredoka+One&family=Varela+Round&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@700&display=swap" rel="stylesheet">

    <style>
        body, html {{ margin: 0; padding: 0; width: 100%; height: 100%; font-family: 'Varela Round', sans-serif; background: #fff; overflow: hidden; }}
        #map {{ width: 100%; height: 100%; background: #ffffff; z-index: 1; }}

        #sidebar-wrapper {{ position: absolute; top: 0; left: 0; width: 260px; height: 100%; z-index: 2000; transition: transform 0.4s ease; pointer-events: none; }}
        #sidebar-wrapper.closed {{ transform: translateX(-260px); }}

        #sidebar {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: #f4f4f4; box-shadow: 2px 0 10px rgba(0,0,0,0.1); padding: 0; overflow-y: auto; pointer-events: auto; }}
        #sidebar::-webkit-scrollbar {{ width: 0px; background: transparent; }}

        #toggle-btn {{ position: absolute; top: 20px; right: -25px; width: 25px; height: 40px; background: #111; color: white; border: none; font-size: 18px; cursor: pointer; border-radius: 0 15px 15px 0; font-weight: bold; display: flex; align-items: center; justify-content: center; box-shadow: 3px 0 5px rgba(0,0,0,0.15); pointer-events: auto; font-family: monospace; }}

        .sidebar-header {{ background: #111; color: white; padding: 15px 20px; font-family: 'Fredoka One', cursive; font-size: 22px; line-height: 1.1; border-bottom-right-radius: 15px; margin-bottom: 15px; letter-spacing: 0.5px; }}
        .sidebar-content {{ padding: 0 20px 20px 20px; display: flex; flex-direction: column; gap: 10px; }}

        .pill-btn {{ width: 100%; padding: 10px; background: white; border: 2px dashed #111; border-radius: 20px; color: #111; font-weight: bold; cursor: pointer; transition: 0.2s; font-family: 'Fredoka One', cursive; font-size: 14px; text-transform: uppercase; }}
        .pill-btn.active {{ background: #111; color: white; border-style: solid; }}

        .white-box {{ background: white; border-radius: 25px; padding: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); position: relative; margin-top: 5px; text-align: center; }}

        .box-title {{ font-family: 'Fredoka One', cursive; font-size: 24px; margin: 10px 0 5px 0; text-transform: uppercase; }}
        .title-blue {{ color: #0f8cc6; }}
        .title-orange {{ color: #f2623a; }}
        .desc {{ font-size: 11px; color: #000; line-height: 1.3; margin-top: 0; margin-bottom: 10px; font-weight: bold; }}

        .mat-container {{ display: flex; flex-direction: column; align-items: flex-start; gap: 5px; margin-bottom: 10px; padding-left: 10px; }}
        .mat-row {{ display: flex; align-items: center; cursor: pointer; transition: 0.2s; position: relative; width: 100%; }}
        .mat-row:hover {{ transform: translateX(5px); }}
        .mat-row.inactive {{ opacity: 0.3; filter: grayscale(100%); }}

        .mat-icon-wrapper {{ width: 45px; height: 25px; position: relative; margin-right: 15px; border-radius: 6px; overflow: hidden; border: 1px solid #333; background: #fff; flex-shrink:0; }}

        .icon-tugla-bg {{ width: 100%; height: 100%; background: linear-gradient(90deg, #333 1px, transparent 1px) 0 0, linear-gradient(#333 1px, transparent 1px) 0 0; background-size: 10px 8px; position: absolute; z-index: 2; }}
        .icon-ahsap-bg {{ width: 100%; height: 100%; background: repeating-linear-gradient(180deg, transparent, transparent 6px, #333 6px, #333 7px); position: absolute; z-index: 2; }}
        .icon-sivali-bg {{ width: 100%; height: 100%; background: radial-gradient(#333 1px, transparent 1px); background-size: 6px 6px; position: absolute; z-index: 2; }}

        .mat-text {{ font-size: 14px; color: #333; font-family: 'Varela Round', sans-serif; z-index: 10; position: relative; font-weight: normal; }}

        .color-graphic {{ width: 100%; height: 120px; background: #fff; border-radius: 15px; position: relative; overflow: hidden; margin-bottom: 5px; display: grid; grid-template-columns: repeat(5, 1fr); grid-template-rows: repeat(5, 1fr); gap: 0; }}
        .color-graphic::before {{ content: ''; position: absolute; top:0; left:0; width:100%; height:100%; border-radius: 15px; pointer-events: none; z-index: 20; box-shadow: inset 0 0 0 2px #fff; }}

        .color-circle-btn {{ width: 150%; height: 150%; border-radius: 50%; transform: translate(-15%, -15%); transition: 0.2s ease-in-out; cursor: pointer; border: none; padding: 0; position: relative; z-index: 10; }}
        .color-circle-btn:hover {{ transform: translate(-15%, -15%) scale(1.2); z-index: 15; box-shadow: 0 0 5px rgba(0,0,0,0.5); }}

        .dominant-color-container {{ display: flex; align-items: center; justify-content: center; gap: 10px; margin-top: 15px; cursor: pointer; transition: 0.2s; padding-bottom: 5px; }}
        .dominant-color-container:hover {{ transform: scale(1.05); }}
        .dominant-circle {{ width: 32px; height: 32px; border-radius: 50%; background: {most_dominant_color}; box-shadow: 0 2px 5px rgba(0,0,0,0.2); flex-shrink: 0; }}
        .dominant-text {{ font-family: 'Varela Round', sans-serif; font-size: 13px; color: #333; font-weight: bold; text-align: left; line-height: 1.1; }}

        .back-to-intro-btn {{ background: transparent; color: #777; border: 1px dashed #ccc; border-radius: 12px; padding: 8px 12px; font-family: 'Varela Round', sans-serif; font-size: 11px; font-weight: bold; cursor: pointer; transition: 0.2s; text-transform: uppercase; margin-top: 15px; width: 100%; letter-spacing: 0.5px; }}
        .back-to-intro-btn:hover {{ background: #111; color: #fff; border-color: #111; border-style: solid; }}

        .bottom-tab {{ position: absolute; bottom: 85px; left: 260px; background: #111; color: white; font-family: 'Fredoka One', cursive; font-size: 12px; padding: 8px 15px; border-radius: 0 15px 0 0; box-shadow: 2px -2px 5px rgba(0,0,0,0.1); z-index: 1500; cursor: pointer; transition: transform 0.4s ease, bottom 0.4s ease, left 0.4s ease; letter-spacing: 0.5px; }}
        #sidebar-wrapper.closed ~ .bottom-tab {{ left: 0px; }}

        #bottombar {{ position: absolute; bottom: 0; left: 260px; width: calc(100% - 260px); height: 85px; background: #f5f5f5; z-index: 1000; display: flex; align-items: center; padding: 5px 15px; overflow-x: auto; gap: 12px; white-space: nowrap; transition: transform 0.4s ease, left 0.4s ease, width 0.4s ease; box-shadow: 0 -2px 10px rgba(0,0,0,0.05); border-radius: 0 20px 0 0; }}
        #bottombar::-webkit-scrollbar {{ height: 8px; }}
        #bottombar::-webkit-scrollbar-track {{ background: #e0e0e0; border-radius: 4px; margin: 0 15px; }}
        #bottombar::-webkit-scrollbar-thumb {{ background: #c0c0c0; border-radius: 4px; }}
        #bottombar::-webkit-scrollbar-thumb:hover {{ background: #a0a0a0; }}

        #sidebar-wrapper.closed ~ #bottombar {{ left: 0; width: 100%; border-radius: 0; }}
        #bottombar.closed {{ transform: translateY(85px); }}
        .bottom-tab.closed {{ transform: translateY(85px); }}

        .bottom-img {{ height: 60px; width: auto; object-fit: contain; filter: drop-shadow(2px 2px 3px rgba(0,0,0,0.2)); transition: 0.2s; cursor: pointer; flex-shrink: 0; border-radius: 4px; }}
        .bottom-img:hover {{ transform: translateY(-3px) scale(1.05); filter: drop-shadow(2px 4px 6px rgba(0,0,0,0.3)); }}

        .selected-card {{ display: flex; align-items: center; background: white; padding: 5px 12px; border-radius: 12px; border: 1px solid #ddd; min-width: 200px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); flex-shrink: 0; transition: 0.2s; cursor: pointer; margin-right: 5px; }}
        .selected-card:hover {{ transform: translateY(-3px); box-shadow: 0 4px 8px rgba(0,0,0,0.1); background: #fdfdfd; }}
        .selected-card img {{ height: 60px; width: auto; object-fit: contain; margin-right: 12px; border-radius: 4px; }}
        .selected-card-info {{ font-family: 'Varela Round', sans-serif; font-size: 11px; color: #555; line-height: 1.4; display: flex; flex-direction: column; }}
        .selected-card-info b {{ color: #111; font-weight: bold; font-family: 'Fredoka One', cursive; letter-spacing: 0.5px; margin-right: 3px; }}

        .custom-facade-icon {{ background: transparent; border: none; cursor: pointer; }}
        .custom-facade-icon:hover img {{ transform: scale(1.1); filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.8)); }}

        .special-landmark-icon {{ background: transparent; border: none; cursor: pointer; z-index: 1000 !important; }}
        .special-landmark-icon img {{ transform: scale(1.0); filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.8)); transition: 0.2s; }}
        .special-landmark-icon:hover img {{ transform: scale(1.15); filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.9)); }}

        #info-box {{ position: absolute; top: 20px; right: 20px; background: rgba(0,0,0,0.7); color: white; padding: 10px 15px; border-radius: 8px; z-index: 1000; font-size: 12px; font-weight: bold; font-family: monospace; pointer-events: none; }}

        #facade-btn-container {{ position: absolute; top: 60px; right: 20px; z-index: 3500; display: flex; align-items: center; background: white; border-radius: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); cursor: pointer; transition: 0.2s; overflow: hidden; border: 1px solid rgba(0,0,0,0.05); }}
        #facade-btn-container:hover {{ transform: scale(1.05); box-shadow: 0 6px 15px rgba(0,0,0,0.25); }}

        #abstract-btn-container {{ position: absolute; top: 115px; right: 20px; z-index: 3500; display: flex; align-items: center; background: white; border-radius: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.15); cursor: pointer; transition: 0.2s; overflow: hidden; border: 1px solid rgba(0,0,0,0.05); }}
        #abstract-btn-container:hover {{ transform: scale(1.05); box-shadow: 0 6px 15px rgba(0,0,0,0.25); }}

        .facade-btn-text {{ color: #111; font-family: 'Fredoka One', cursive; font-size: 14px; padding: 10px 15px 10px 20px; letter-spacing: 0.5px; }}
        .facade-btn-icon {{ background: #111; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-size: 18px; font-weight: bold; margin-left: -5px; }}

        #collage-overlay {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: rgba(245, 245, 245, 0.98); z-index: 3000; visibility: hidden; opacity: 0; transition: opacity 0.4s ease; overflow: hidden; cursor: grab; }}
        #collage-overlay.active {{ visibility: visible; opacity: 1; }}
        #collage-content {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; transform-origin: 0 0; will-change: transform; transition: transform 0.1s ease-out; }}

        #collage-colorbar {{ position: absolute; bottom: 0; left: 0; width: 100%; height: 35px; display: flex; z-index: 3500; opacity: 0; transition: opacity 0.8s ease; pointer-events: none; }}

        .flying-facade {{ position: absolute; top: 0; left: 0; transition: transform 1.2s cubic-bezier(0.25, 1, 0.5, 1), opacity 0.3s; object-fit: contain; object-position: bottom center; filter: drop-shadow(2px 2px 4px rgba(0,0,0,0.3)); z-index: 3001; will-change: transform; backface-visibility: hidden; }}

        .flying-facade-label {{ position: absolute; transform-origin: left center; transform: rotate(-90deg); color: #e46a39; font-family: 'Space Grotesk', sans-serif; font-size: 14px; letter-spacing: 2px; white-space: nowrap; opacity: 0; transition: opacity 0.8s ease; pointer-events: none; z-index: 3002; }}

        /* GİRİŞ EKRANI */
        #intro-screen {{ position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: #000; z-index: 9999; display: flex; align-items: center; justify-content: center; overflow: hidden; cursor: pointer; transition: opacity 1.5s ease, background 1.5s ease; }}
        #intro-background {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; pointer-events: none; }}
        .radial-item {{ position: absolute; background-size: cover; background-position: center; box-shadow: 0 2px 6px rgba(0,0,0,0.6); border-radius: 4px; transition: transform 1.5s cubic-bezier(0.25, 1, 0.5, 1), opacity 1.3s ease; will-change: transform, opacity; transform-origin: center center; filter: saturate(1.2) brightness(1.1); border: 1px solid rgba(255,255,255,0.15); }}
        #intro-screen.disperse {{ background: transparent !important; pointer-events: none; }}
        #intro-screen.disperse .radial-item {{ opacity: 0 !important; }}
        .intro-center-content {{ position: relative; z-index: 10000; text-align: center; pointer-events: none; transition: opacity 0.8s ease, transform 0.8s ease; max-width: 50%; text-shadow: 0 4px 10px rgba(0,0,0,0.8); }}
        .intro-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.2vw; color: #fff; margin: 0; line-height: 1.2; letter-spacing: 4px; text-transform: uppercase; }}
        #intro-screen.disperse .intro-center-content {{ opacity: 0; transform: scale(0.5); }}

        .intro-footer-text {{ position: absolute; bottom: 40px; right: 40px; color: white; text-align: right; z-index: 10000; font-family: 'Varela Round', sans-serif; pointer-events: none; transition: opacity 0.8s ease; text-shadow: 0 2px 5px rgba(0,0,0,0.8); }}
        #intro-screen.disperse .intro-footer-text {{ opacity: 0; }}
    </style>
</head>
<body>
    <div id="intro-screen" onclick="closeIntro()">
        <div id="intro-background"></div>
        <div class="intro-center-content"><h1 class="intro-title">FACADE SPECTRUM<br>OF BALAT</h1></div>

        <div class="intro-footer-text">
            <div style="font-weight:bold; font-size:16px; margin-bottom:5px;">The outlier colors found in Balat.</div>
            <div style="font-size:13px; opacity:0.8;">Click anywhere to view all buildings...</div>
        </div>
    </div>

    <div id="sidebar-wrapper">
        <button id="toggle-btn" onclick="toggleSidebar()">‹</button>
        <div id="sidebar">
            <div class="sidebar-header">Characteristic<br>of BALAT</div>
            <div class="sidebar-content">
                <button class="pill-btn active" onclick="resetFilters(this)">ALL BUILDINGS</button>
                <button class="pill-btn" style="background: transparent; color: #111;">SELECTIONS</button>
                <div class="white-box">
                    <div class="mat-container">
                        <div class="mat-row" onclick="filterMaterial('Tuğla', this)"><div class="mat-icon-wrapper"><div class="icon-tugla-bg"></div></div><span class="mat-text">brick (tuğla)</span></div>
                        <div class="mat-row" onclick="filterMaterial('Ahşap', this)"><div class="mat-icon-wrapper"><div class="icon-ahsap-bg"></div></div><span class="mat-text">wood (ahşap)</span></div>
                        <div class="mat-row" onclick="filterMaterial('Sıvalı', this)"><div class="mat-icon-wrapper"><div class="icon-sivali-bg"></div></div><span class="mat-text">plaster (sıvalı)</span></div>
                    </div>
                    <h3 class="box-title title-blue">MATERIAL</h3><p class="desc">Choose the buildings<br>that have the <b>material</b><br>you want to see.</p>
                </div>
                <div class="white-box">
                    <div class="color-graphic" id="color-graphic"></div>
                    <h3 class="box-title title-orange">COLOR</h3><p class="desc">Choose the buildings in<br>the <b>color</b> you want to<br>see.</p>
                </div>
                <div class="dominant-color-container" onclick="filterColor('{most_dominant_color}')">
                    <div class="dominant-circle"></div><div class="dominant-text">the most<br>dominant color</div>
                </div>

                <button class="back-to-intro-btn" onclick="openIntro()">↺ Back to intro screen</button>
            </div>
        </div>
    </div>

    <div class="bottom-tab" id="bottom-tab" onclick="toggleBottomBar()">Selected facades</div>
    <div id="bottombar"></div>
    <div id="info-box">Architectural Symbolic Facades</div>

    <div id="facade-btn-container" onclick="toggleCollage('all')">
        <div class="facade-btn-text">click for all facade</div><div class="facade-btn-icon">▶</div>
    </div>

    <div id="abstract-btn-container" onclick="toggleCollage('abstract')">
        <div class="facade-btn-text" style="color: #0f8cc6;">click for symbolic facades</div><div class="facade-btn-icon" style="background: #0f8cc6;">▶</div>
    </div>

    <div id="collage-overlay">
        <div id="collage-content"></div>
        <div id="collage-colorbar"></div>
    </div>

    <div id="map"></div>

    <script>
        const minLat = {min_lat} - 0.002, maxLat = {max_lat} + 0.002;
        const minLon = {min_lon} - 0.002, maxLon = {max_lon} + 0.002;
        const bounds = [[minLat, minLon], [maxLat, maxLon]];

        const map = L.map('map', {{ zoomControl: false, minZoom: 16, maxZoom: 19, maxBounds: bounds, maxBoundsViscosity: 1.0, wheelPxPerZoomLevel: 120 }}).setView([41.0315, 28.9480], 16);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_nolabels/{{z}}/{{x}}/{{y}}{{r}}.png', {{ attribution: '', maxZoom: 22, maxNativeZoom: 18 }}).addTo(map);

        map.createPane('yollarPane'); map.getPane('yollarPane').style.zIndex = 700; map.getPane('yollarPane').style.pointerEvents = 'none';
        map.createPane('sinirPane'); map.getPane('sinirPane').style.zIndex = 800; map.getPane('sinirPane').style.pointerEvents = 'none';
        map.createPane('heatmapPane'); map.getPane('heatmapPane').style.zIndex = 450; map.getPane('heatmapPane').style.pointerEvents = 'none';

        const geojsonData = {geojson_data};
        const balatSiniri = {balat_geojson};
        const topColors = {palet_json};
        const yollarData = {yollar_geojson};
        const manuelPinsData = {manuel_pin_json};

        const worldCoords = [[-90, -180], [90, -180], [90, 180], [-90, 180]];
        const balatCoords = balatSiniri.features[0].geometry.coordinates[0];
        L.polygon([worldCoords, balatCoords], {{ color: 'transparent', fillColor: '#e5e5e5', fillOpacity: 0.8 }}).addTo(map);

        L.geoJSON(balatSiniri, {{ pane: 'sinirPane', style: {{ color: '#111111', weight: 2, fillOpacity: 0, dashArray: '5, 5' }} }}).addTo(map);
        L.geoJSON(yollarData, {{ pane: 'yollarPane', style: {{ color: '#ffffff', weight: 3, opacity: 1.0 }}, pointToLayer: (feature, latlng) => L.circleMarker(latlng, {{radius: 0, opacity: 0, fillOpacity: 0}}) }}).addTo(map);

        const colorGraphic = document.getElementById('color-graphic');
        topColors.forEach((color) => {{
            const btn = document.createElement('button');
            btn.className = 'color-circle-btn'; btn.style.backgroundColor = color; btn.setAttribute('data-color', color);
            btn.onclick = () => filterColor(color); colorGraphic.appendChild(btn);
        }});

        let activeMaterial = 'Tümü', activeColor = 'Tümü', semanticStep = 0, isBottomBarOpen = true;
        let selectedFeatures = [];

        window.removeBuilding = function(id) {{ selectedFeatures = selectedFeatures.filter(f => f.id !== id); renderBottomBar(); }};
        window.selectBuilding = function(props) {{ if (!selectedFeatures.some(f => f.id === props.id)) selectedFeatures.unshift(props); renderBottomBar(); if(!isBottomBarOpen) toggleBottomBar(); setTimeout(() => document.getElementById('bottombar').scrollLeft = 0, 50); }};
        window.selectBuildingFromIcon = function(el, id, img, sokak, kat, zemin, malzeme) {{ selectBuilding({{ id: id, img: img, sokak: sokak, kat: kat, zemin: zemin, malzeme: malzeme }}); }};

        function renderBottomBar() {{
            const bottomBar = document.getElementById('bottombar'); bottomBar.innerHTML = '';
            selectedFeatures.forEach(props => {{
                const sSokak = (props.sokak || "Bilinmiyor").replace(/'/g, ""); const sKat = (props.kat || "").replace(/'/g, ""); const sZemin = (props.zemin || "").replace(/'/g, ""); const sMalzeme = (props.malzeme || "").replace(/'/g, "");
                bottomBar.innerHTML += `<div class="selected-card" onclick="removeBuilding('${{props.id}}')"><img loading="lazy" src="${{props.img}}"><div class="selected-card-info"><span><b>Street:</b> ${{sSokak}}</span><span><b>Floors:</b> ${{sKat}}</span><span><b>Ground:</b> ${{sZemin}}</span><span><b>Material:</b> ${{sMalzeme}}</span></div></div>`;
            }});

            let visibleCount = 0; const viewBounds = map.getBounds();
            geojsonData.features.forEach(feature => {{
                let props = feature.properties; let isHidden = false;
                if (activeMaterial !== 'Tümü' && props.malzeme !== activeMaterial) isHidden = true;
                if (activeColor !== 'Tümü' && props.c25_closest !== activeColor) isHidden = true;
                const latLng = L.latLng(props.center_lat, props.center_lon);
                if (!isHidden && viewBounds.contains(latLng) && visibleCount < 100) {{
                    if (!selectedFeatures.some(f => f.id === props.id)) {{
                        const sSokak = (props.sokak || "Bilinmiyor").replace(/'/g, ""); const sKat = (props.kat || "").replace(/'/g, ""); const sZemin = (props.zemin || "").replace(/'/g, ""); const sMalzeme = (props.malzeme || "").replace(/'/g, "");
                        bottomBar.innerHTML += `<img class="bottom-img" loading="lazy" src="${{props.img}}" onclick="selectBuildingFromIcon(this, '${{props.id}}', '${{props.img}}', '${{sSokak}}', '${{sKat}}', '${{sZemin}}', '${{sMalzeme}}')">`; visibleCount++;
                    }}
                }}
            }});
            if ((activeMaterial !== 'Tümü' || activeColor !== 'Tümü') && !isBottomBarOpen && (visibleCount > 0 || selectedFeatures.length > 0)) toggleBottomBar();
        }}

        let poligonLayer = L.geoJSON(geojsonData, {{ pane: 'heatmapPane', style: (feature) => ({{ fillColor: 'transparent', color: 'transparent' }}) }}).addTo(map);
        let iconLayer = L.layerGroup().addTo(map);

        map.getContainer().addEventListener('wheel', function(e) {{
            const atMinZoom = (map.getZoom() <= map.getMinZoom() + 0.05);
            if (semanticStep > 0) {{
                e.preventDefault(); e.stopPropagation(); if (map.scrollWheelZoom.enabled()) map.scrollWheelZoom.disable();
                if (e.deltaY > 0) {{ if (semanticStep < 5) {{ semanticStep++; updateMap(); }} }}
                else if (e.deltaY < 0) {{ semanticStep--; updateMap(); if (semanticStep === 0) map.scrollWheelZoom.enable(); }}
                return;
            }}
            if (semanticStep === 0 && atMinZoom) {{
                if (e.deltaY > 0) {{ e.preventDefault(); e.stopPropagation(); map.scrollWheelZoom.disable(); semanticStep++; updateMap(); return; }}
            }}
            if (semanticStep === 0 && !map.scrollWheelZoom.enabled()) map.scrollWheelZoom.enable();
        }}, {{passive: false}});

        function updateMap() {{
            const z = map.getZoom(); const viewBounds = map.getBounds(); iconLayer.clearLayers();
            const scaleFactor = Math.pow(2, z - 18);

            let status = "";
            if (z > 16) status = "Architectural Symbolic Facades";
            else {{
                if (semanticStep === 0) status = "Architectural Symbolic Facades";
                else if (semanticStep === 1) status = "Facade Color";
                else if (semanticStep === 2) status = "Ten Dominant Color";
                else if (semanticStep === 3) status = "Five Dominant Color";
                else if (semanticStep === 4) status = "Three Dominant Color";
                else if (semanticStep === 5) status = "The Dominant Color of Balat";
            }}
            document.getElementById('info-box').innerText = status;

            poligonLayer.setStyle(function(feature) {{
                let props = feature.properties; let isHidden = false;
                if (activeMaterial !== 'Tümü' && props.malzeme !== activeMaterial) isHidden = true;
                if (activeColor !== 'Tümü' && props.c25_closest !== activeColor) isHidden = true;

                let fillColor = props.renk_orj, strokeColor = '#ffffff', fillOp = isHidden ? 0.05 : 0.95, strokeOp = 0, weight = 0;
                if (z > 16 || semanticStep === 0) return {{ fillColor: fillColor, fillOpacity: 0.0, color: 'transparent', weight: 0 }};

                if (semanticStep >= 1) {{
                    if (semanticStep === 2) fillColor = props.c10; if (semanticStep === 3) fillColor = props.c5;
                    if (semanticStep === 4) fillColor = props.c3; if (semanticStep === 5) fillColor = props.c1;
                    strokeColor = fillColor; weight = semanticStep * 15; fillOp = isHidden ? 0.0 : 0.85; strokeOp = isHidden ? 0.0 : 0.4;
                }}
                return {{ fillColor: fillColor, fillOpacity: fillOp, color: strokeColor, weight: weight, opacity: strokeOp }};
            }});

            geojsonData.features.forEach(feature => {{
                let props = feature.properties; let isHidden = false;
                if (activeMaterial !== 'Tümü' && props.malzeme !== activeMaterial) isHidden = true;
                if (activeColor !== 'Tümü' && props.c25_closest !== activeColor) isHidden = true;
                const latLng = L.latLng(props.center_lat, props.center_lon);

                if ((z > 16 || semanticStep === 0) && !isHidden) {{
                    let h = Math.max(28, 45 * scaleFactor);
                    const iconHtml = `<div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; transition:0.2s;"><img loading="lazy" src="${{props.img}}" style="height:${{h}}px; width:auto; object-fit:contain; filter: drop-shadow(1px 1px 1px rgba(0,0,0,0.5)); transition:0.2s;"></div>`;
                    L.marker([props.center_lat, props.center_lon], {{ icon: L.divIcon({{ html: iconHtml, className: 'custom-facade-icon', iconSize: [12, 24], iconAnchor: [6, 12] }}) }}).addTo(iconLayer);
                }}
            }});

            if (z > 16 || semanticStep === 0) {{
                manuelPinsData.forEach(pin => {{
                    if (pin.b64 && pin.b64 !== "") {{
                        if (activeMaterial === 'Tümü' && activeColor === 'Tümü') {{
                            let lh = Math.max(38, 65 * scaleFactor);
                            const iconHtml = `<div style="width:100%; height:100%; display:flex; justify-content:center; align-items:center; transition:0.2s;"><img loading="lazy" src="${{pin.b64}}" style="height:${{lh}}px; width:auto; object-fit:contain; filter: drop-shadow(1px 1px 2px rgba(0,0,0,0.8)); transition:0.2s;"></div>`;
                            let marker = L.marker([pin.lat, pin.lon], {{ icon: L.divIcon({{ html: iconHtml, className: 'special-landmark-icon', iconSize: [12, 24], iconAnchor: [6, 12] }}) }}).addTo(iconLayer);
                            marker.on('click', () => selectBuilding({{ id: "manuel_" + pin.isim, img: pin.b64, sokak: pin.isim, kat: "Sembolik", zemin: "Sembolik", malzeme: "Özel" }}));
                        }}
                    }}
                }});
            }}
            renderBottomBar();
        }}
        map.on('zoomend', updateMap); map.on('moveend', updateMap);

        function toggleSidebar() {{
            const wrapper = document.getElementById('sidebar-wrapper'), btn = document.getElementById('toggle-btn');
            if(wrapper.classList.contains('closed')) {{ wrapper.classList.remove('closed'); btn.innerHTML = '‹'; }} else {{ wrapper.classList.add('closed'); btn.innerHTML = '›'; }}
        }}
        function toggleBottomBar() {{
            const bar = document.getElementById('bottombar'), tab = document.getElementById('bottom-tab');
            if (isBottomBarOpen) {{ bar.classList.add('closed'); tab.classList.add('closed'); isBottomBarOpen = false; }} else {{ bar.classList.remove('closed'); tab.classList.remove('closed'); isBottomBarOpen = true; }}
        }}

        function filterMaterial(malzeme, element) {{
            if (activeMaterial === malzeme) {{
                activeMaterial = 'Tümü';
                document.querySelectorAll('.mat-row').forEach(el => el.classList.remove('inactive'));
            }} else {{
                activeMaterial = malzeme;
                document.querySelectorAll('.mat-row').forEach(el => {{
                    if(el === element) el.classList.remove('inactive');
                    else el.classList.add('inactive');
                }});
            }}
            updateSelectionState();
            updateMap();
        }}

        function filterColor(renk) {{
            if (activeColor === renk) {{
                activeColor = 'Tümü';
                document.querySelectorAll('.color-circle-btn').forEach(el => {{ el.style.opacity = '1'; el.style.transform = 'translate(-15%, -15%) scale(1)'; el.style.zIndex = '10'; }});
            }} else {{
                activeColor = renk;
                document.querySelectorAll('.color-circle-btn').forEach(el => {{
                    if (el.getAttribute('data-color') === renk) {{ el.style.opacity = '1'; el.style.transform = 'translate(-15%, -15%) scale(1.1)'; el.style.zIndex = '15'; }}
                    else {{ el.style.opacity = '0.15'; el.style.transform = 'translate(-15%, -15%) scale(1)'; el.style.zIndex = '10'; }}
                }});
            }}
            updateSelectionState();
            updateMap();
        }}

        function updateSelectionState() {{
            const btn = document.querySelectorAll('.pill-btn')[0];
            if (activeMaterial === 'Tümü' && activeColor === 'Tümü') {{
                btn.classList.add('active');
            }} else {{
                btn.classList.remove('active');
            }}
        }}

        function resetFilters(btnElement) {{
            activeMaterial = 'Tümü'; activeColor = 'Tümü';
            document.querySelectorAll('.pill-btn').forEach(el => el.classList.remove('active')); btnElement.classList.add('active');
            document.querySelectorAll('.mat-row').forEach(el => el.classList.remove('inactive'));
            document.querySelectorAll('.color-circle-btn').forEach(el => {{ el.style.opacity = '1'; el.style.transform = 'translate(-15%, -15%) scale(1)'; el.style.zIndex = '10'; }});
            updateMap();
        }}

        let isCollageMode = false, activeCollageType = null, flyingElements = [], collageZoom = 1, collagePanX = 0, collagePanY = 0, isDraggingCollage = false, startDragX, startDragY;
        const overlay = document.getElementById('collage-overlay'), cContent = document.getElementById('collage-content');

        overlay.addEventListener('wheel', (e) => {{
            if (!isCollageMode) return; e.preventDefault();
            let zoomDelta = e.deltaY < 0 ? 1.15 : 0.85; let newZoom = collageZoom * zoomDelta;
            if (newZoom < 1.0) newZoom = 1.0; if (newZoom > 15) newZoom = 15;
            collagePanX = e.clientX - (e.clientX - collagePanX) * (newZoom / collageZoom); collagePanY = e.clientY - (e.clientY - collagePanY) * (newZoom / collageZoom);
            collageZoom = newZoom; cContent.style.transition = 'none'; cContent.style.transform = `translate3d(${{collagePanX}}px, ${{collagePanY}}px, 0) scale(${{collageZoom}})`;
        }});
        overlay.addEventListener('mousedown', (e) => {{ if (!isCollageMode) return; isDraggingCollage = true; startDragX = e.clientX - collagePanX; startDragY = e.clientY - collagePanY; overlay.style.cursor = 'grabbing'; }});
        window.addEventListener('mousemove', (e) => {{ if (!isDraggingCollage) return; collagePanX = e.clientX - startDragX; collagePanY = e.clientY - startDragY; cContent.style.transition = 'none'; cContent.style.transform = `translate3d(${{collagePanX}}px, ${{collagePanY}}px, 0) scale(${{collageZoom}})`; }});
        window.addEventListener('mouseup', () => {{ isDraggingCollage = false; overlay.style.cursor = 'grab'; }});

        function hexToHSL(H) {{
            if (!H) return {{h: 0, l: 0}};
            let r = 0, g = 0, b = 0;
            if (H.length == 4) {{ r = "0x" + H[1] + H[1]; g = "0x" + H[2] + H[2]; b = "0x" + H[3] + H[3]; }}
            else if (H.length == 7) {{ r = "0x" + H[1] + H[2]; g = "0x" + H[3] + H[4]; b = "0x" + H[5] + H[6]; }}
            r /= 255; g /= 255; b /= 255;
            let cmin = Math.min(r,g,b), cmax = Math.max(r,g,b), delta = cmax - cmin, h = 0, s = 0, l = 0;
            if (delta == 0) h = 0;
            else if (cmax == r) h = ((g - b) / delta) % 6;
            else if (cmax == g) h = (b - r) / delta + 2;
            else h = (r - g) / delta + 4;
            h = Math.round(h * 60); if (h < 0) h += 360;
            l = (cmax + cmin) / 2;
            return {{h, l}};
        }}

        function toggleCollage(type) {{
            if (isCollageMode && activeCollageType !== type) return;

            const btnContainer = type === 'all' ? document.getElementById('facade-btn-container') : document.getElementById('abstract-btn-container');
            const otherBtnContainer = type === 'all' ? document.getElementById('abstract-btn-container') : document.getElementById('facade-btn-container');
            const btnText = btnContainer.querySelector('.facade-btn-text');
            const btnIcon = btnContainer.querySelector('.facade-btn-icon');

            isCollageMode = !isCollageMode;

            if (isCollageMode) {{
                activeCollageType = type;
                otherBtnContainer.style.display = 'none';

                overlay.classList.add('active');
                btnText.innerText = "back to map";
                btnIcon.innerText = "×";
                document.querySelectorAll('.custom-facade-icon').forEach(el => el.style.opacity = '0');
                document.querySelectorAll('.special-landmark-icon').forEach(el => el.style.opacity = '0');

                let activeFeatures = [];
                if (type === 'all') {{
                    activeFeatures = [...geojsonData.features];
                    manuelPinsData.forEach(pin => {{
                        if (pin.b64 && pin.b64 !== "") {{
                            activeFeatures.push({{ properties: {{ img: pin.b64, en_isim: pin.en_isim, center_lat: pin.lat, center_lon: pin.lon, renk_orj: pin.renk_orj || '#888888' }} }});
                        }}
                    }});
                }} else if (type === 'abstract') {{
                    manuelPinsData.forEach(pin => {{
                        if (pin.b64 && pin.b64 !== "") {{
                            activeFeatures.push({{ properties: {{ img: pin.b64, en_isim: pin.en_isim, center_lat: pin.lat, center_lon: pin.lon, renk_orj: pin.renk_orj || '#888888' }} }});
                        }}
                    }});
                }}

                activeFeatures.sort((a, b) => {{
                    let hslA = hexToHSL(a.properties.renk_orj);
                    let hslB = hexToHSL(b.properties.renk_orj);
                    if (Math.abs(hslA.h - hslB.h) > 15) return hslA.h - hslB.h;
                    return hslB.l - hslA.l;
                }});

                const ww = window.innerWidth, wh = window.innerHeight;
                let totalBuildings = activeFeatures.length;
                let imgW = 24, imgH = 48;

                if (type === 'abstract') {{
                    imgW = Math.floor(ww / totalBuildings);
                    if (imgW > 250) imgW = 250;
                    if (imgW < 60) imgW = 60;
                }}

                let cols = Math.floor(ww / imgW);
                if (cols > totalBuildings) cols = totalBuildings;
                if (cols < 1) cols = 1;

                let rowsCount = Math.ceil(totalBuildings / cols);

                if (type === 'abstract') {{
                    imgH = Math.min(300, Math.floor((wh - 120) / rowsCount));
                }}

                let startX = (ww - (cols * imgW)) / 2;
                let baseHeight = Math.floor(totalBuildings / cols), remainder = totalBuildings % cols;

                let colCapacities = new Array(cols).fill(baseHeight);
                for(let k=0; k<remainder; k++) {{
                    let mid = Math.floor(cols/2); let offset = k % 2 === 0 ? k/2 : -Math.ceil(k/2);
                    colCapacities[(mid + offset + cols) % cols]++;
                }}

                if (type === 'all') {{
                    for (let k = 0; k < Math.floor(totalBuildings * 0.1); k++) {{
                        let from = Math.floor(Math.random() * cols), to = Math.floor(Math.random() * cols);
                        if (colCapacities[from] > 1) {{ colCapacities[from]--; colCapacities[to]++; }}
                    }}
                }}

                collageZoom = 1; collagePanX = 0; collagePanY = 0;
                cContent.style.transition = 'transform 1.2s cubic-bezier(0.25, 1, 0.5, 1)';
                cContent.style.transform = `translate3d(${{collagePanX}}px, ${{collagePanY}}px, 0) scale(${{collageZoom}})`;

                let currentFeatureIdx = 0;
                for (let c = 0; c < cols; c++) {{
                    let capacity = colCapacities[c];
                    for (let r = 0; r < capacity; r++) {{
                        if (currentFeatureIdx >= totalBuildings) break;
                        let f = activeFeatures[currentFeatureIdx];
                        let pt = map.latLngToContainerPoint([f.properties.center_lat, f.properties.center_lon]);

                        let img = document.createElement('img'); img.src = f.properties.img; img.className = 'flying-facade';
                        let startTransformX = pt.x - collagePanX, startTransformY = pt.y - collagePanY, startScale = 12 / imgW;

                        img.style.transform = `translate3d(${{startTransformX}}px, ${{startTransformY}}px, 0) scale(${{startScale}})`;
                        img.style.width = imgW + 'px'; img.style.height = imgH + 'px'; cContent.appendChild(img);

                        let targetX = startX + (c * imgW);
                        let targetY = wh - 70 - (r * imgH) - imgH;

                        flyingElements.push({{ el: img, lat: f.properties.center_lat, lon: f.properties.center_lon, tS: startScale, trgX: targetX, trgY: targetY, isLabel: false }});

                        if (type === 'abstract') {{
                            let label = document.createElement('div');
                            label.className = 'flying-facade-label';
                            label.innerText = f.properties.en_isim || 'Unknown';

                            let labelTargetX = targetX + (imgW / 2);
                            let labelTargetY = targetY - 10;

                            label.style.left = labelTargetX + 'px';
                            label.style.top = labelTargetY + 'px';

                            cContent.appendChild(label);
                            flyingElements.push({{ el: label, isLabel: true }});

                            setTimeout(() => {{ label.style.opacity = '1'; }}, 1000 + (currentFeatureIdx * 50));
                        }}

                        currentFeatureIdx++;
                    }}
                }}

                flyingElements.forEach((item, i) => {{
                    if (!item.isLabel) {{
                        setTimeout(() => {{ item.el.style.transform = `translate3d(${{item.trgX}}px, ${{item.trgY}}px, 0) scale(1)`; }}, 50 + (i * 2));
                    }}
                }});

                let colorBar = document.getElementById('collage-colorbar');
                colorBar.innerHTML = '';
                activeFeatures.forEach(f => {{
                    let cDiv = document.createElement('div');
                    cDiv.style.flex = '1';
                    cDiv.style.backgroundColor = f.properties.renk_orj;
                    colorBar.appendChild(cDiv);
                }});
                setTimeout(() => {{ colorBar.style.opacity = '1'; }}, 1000);

            }} else {{
                btnText.innerText = type === 'all' ? "click for all facade" : "click for symbolic facades";
                btnIcon.innerText = "▶";
                otherBtnContainer.style.display = 'flex';

                collageZoom = 1; collagePanX = 0; collagePanY = 0;
                cContent.style.transition = 'transform 1.2s cubic-bezier(0.25, 1, 0.5, 1)'; cContent.style.transform = `translate3d(0px, 0px, 0) scale(1)`;
                overlay.style.transition = 'background-color 0.8s ease'; overlay.style.backgroundColor = 'transparent';

                document.getElementById('collage-colorbar').style.opacity = '0';

                flyingElements.forEach((item, i) => {{
                    if (item.isLabel) {{
                        item.el.style.opacity = '0';
                    }} else {{
                        let pt = map.latLngToContainerPoint([item.lat, item.lon]);
                        setTimeout(() => {{ item.el.style.transform = `translate3d(${{pt.x}}px, ${{pt.y}}px, 0) scale(${{item.tS}})`; item.el.style.opacity = '0'; }}, i * 2);
                    }}
                }});

                setTimeout(() => {{
                    document.querySelectorAll('.custom-facade-icon').forEach(el => el.style.opacity = '1');
                    document.querySelectorAll('.special-landmark-icon').forEach(el => el.style.opacity = '1');
                    flyingElements.forEach(item => item.el.remove()); flyingElements = [];
                    overlay.classList.remove('active'); setTimeout(() => {{ overlay.style.backgroundColor = ''; overlay.style.transition = 'opacity 0.4s ease'; }}, 100);
                    activeCollageType = null;
                }}, 1300 + (flyingElements.length * 2));
            }}
        }}

        function initIntroScreen() {{
            const introBg = document.getElementById('intro-background');
            introBg.innerHTML = '';

            const features = geojsonData.features.slice(-160).reverse();
            const N = features.length;

            let r = Math.min(window.innerWidth, window.innerHeight) * 0.38;
            if (window.innerWidth > 800) r = Math.max(r, 230);

            const minSize = 22;
            const maxSize = window.innerWidth > 800 ? 75 : 50;

            features.forEach((feature, i) => {{
                let props = feature.properties;
                let color = props.renk_orj;

                let sizeRatio = Math.pow(1 - (i / N), 1.2);
                let size = minSize + sizeRatio * (maxSize - minSize);

                let theta = Math.PI + (i / N) * 2 * Math.PI;

                const tx = Math.cos(theta) * r;
                const ty = Math.sin(theta) * r;

                const item = document.createElement('div');
                item.className = 'radial-item';
                item.style.backgroundImage = `url(${{props.img}})`;
                item.style.width = `${{size}}px`;
                item.style.height = `${{size * 1.5}}px`;
                item.style.backgroundColor = color;
                item.style.left = `calc(50% - ${{size/2}}px)`;
                item.style.top = `calc(50% - {{(size*1.5)/2}}px)`;
                item.dataset.tx = tx;
                item.dataset.ty = ty;
                item.style.transform = `translate(0px, 0px) scale(0)`;
                introBg.appendChild(item);

                setTimeout(() => {{
                    item.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(1)`;
                }}, 50 + (i * 12));
            }});
        }}

        window.addEventListener('load', initIntroScreen);

        window.closeIntro = function() {{
            const intro = document.getElementById('intro-screen');
            if (intro.classList.contains('disperse')) return;
            intro.classList.add('disperse');
            const items = document.querySelectorAll('.radial-item');
            const dist = Math.max(window.innerWidth, window.innerHeight) * 1.5;
            items.forEach((item) => {{
                const tx = parseFloat(item.dataset.tx); const ty = parseFloat(item.dataset.ty);
                const angle = Math.atan2(ty, tx); const outX = Math.cos(angle) * dist; const outY = Math.sin(angle) * dist;
                item.style.transition = `transform 1s cubic-bezier(0.5, 0, 0.75, 0) ${{Math.random() * 0.1}}s, opacity 0.8s ease-in ${{Math.random() * 0.1}}s`;
                item.style.transform = `translate(${{outX}}px, ${{outY}}px) scale(2)`;
            }});
            setTimeout(() => {{ intro.style.display = 'none'; }}, 1200);
        }};

        window.openIntro = function() {{
            const intro = document.getElementById('intro-screen');
            intro.style.display = 'flex';

            void intro.offsetWidth;

            intro.classList.remove('disperse');

            const items = document.querySelectorAll('.radial-item');
            items.forEach((item, i) => {{
                const tx = parseFloat(item.dataset.tx);
                const ty = parseFloat(item.dataset.ty);

                item.style.transition = 'none';
                item.style.transform = `translate(0px, 0px) scale(0)`;

                setTimeout(() => {{
                    item.style.transition = `transform 1.5s cubic-bezier(0.25, 1, 0.5, 1), opacity 1.3s ease`;
                    item.style.transform = `translate(${{tx}}px, ${{ty}}px) scale(1)`;
                }}, 50 + (i * 12));
            }});
        }};
    </script>
</body>
</html>
"""

with open(CIKTI_HTML, "w", encoding="utf-8") as f:
    f.write(html_icerik)

print("[BİTTİ] HTML dosyası oluşturuldu:", CIKTI_HTML)

with zipfile.ZipFile(ZIP_ISMI, "w", zipfile.ZIP_DEFLATED) as zipf:
    zipf.write(CIKTI_HTML)
    for root, dirs, files_in_dir in os.walk(WEB_IMAGES_DIR):
        for file in files_in_dir:
            file_path = os.path.join(root, file)
            zipf.write(file_path, arcname=file_path)

print("="*50)
print(f"!!! İŞLEM TAMAMLANDI - {ZIP_ISMI} DOSYASINI İNDİRİN !!!")
print("="*50)
print("ÖNEMLİ: Colab'ın eski dosyayı vermemesi için zip ismini '_v7' olarak güncelledim.")
print(f"Lütfen sol taraftan '{ZIP_ISMI}' isimli dosyayı indirip açın ve HTML'i çalıştırın.")
