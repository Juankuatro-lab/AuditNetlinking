import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
import chardet
from urllib.parse import urlparse
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration de la page
st.set_page_config(
    page_title="Audit de netlinking - Création d'une roadmap pour satelliser ta stratégie",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Audit de netlinking - Création d'une roadmap pour satelliser ta stratégie")
st.markdown("**Priorisez vos campagnes de netlinking en croisant les données Ahrefs et GSC**")

def detect_encoding(file_content):
    """Détecte l'encodage d'un fichier"""
    detected = chardet.detect(file_content)
    return detected['encoding'] if detected['confidence'] > 0.7 else 'utf-8'

def read_file_universal(uploaded_file):
    """Lit un fichier CSV ou Excel avec gestion des encodages"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Gestion des fichiers Excel
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
            return df
        
        # Gestion des fichiers CSV (code existant)
        content = uploaded_file.read()
        
        try:
            decoded_content = content.decode('utf-16le')
            decoded_content = decoded_content.replace('\x00', '')
            decoded_content = decoded_content.replace('\ufeff', '')
            df = pd.read_csv(StringIO(decoded_content), sep='\t')
            return df
        except:
            pass
        
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep='\t')
            return df
        except:
            pass
        
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep=',')
            return df
        except:
            pass
        
        encoding = detect_encoding(content)
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding=encoding, sep='\t')
        return df
        
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {str(e)}")
        return None

def clean_percentage(value):
    """Nettoie les pourcentages"""
    if isinstance(value, str):
        return float(value.replace('%', '').replace(',', '.'))
    return value

@st.cache_data
def calculate_thematic_relevance_optimized(domains_series, keywords_data=None, pages_data=None):
    """Version vectorisée et cachée du calcul de pertinence thématique"""
    if keywords_data is None and pages_data is None:
        return pd.Series(0, index=domains_series.index)
    
    relevance_scores = pd.Series(0.0, index=domains_series.index)
    
    keyword_dict = {}
    if keywords_data is not None:
        for _, row in keywords_data.iterrows():
            keyword = str(row.get('Keyword', '')).lower()
            words = set(re.findall(r'\w+', keyword))
            if words:
                keyword_dict[frozenset(words)] = row.get('Search Volume', 0)
    
    pages_dict = {}
    if pages_data is not None:
        for _, row in pages_data.iterrows():
            page_url = str(row.get('Pages les plus populaires', '')).lower()
            words = set(re.findall(r'\w+', page_url))
            if words:
                pages_dict[frozenset(words)] = row.get('Clics', 0)
    
    for idx, domain in domains_series.items():
        domain_words = set(re.findall(r'\w+', str(domain).lower()))
        score = 0
        
        for keyword_words, volume in keyword_dict.items():
            common_words = domain_words & keyword_words
            if common_words:
                score += len(common_words) * volume / 1000
        
        for page_words, clics in pages_dict.items():
            common_words = domain_words & page_words
            if common_words:
                score += len(common_words) * clics / 100
        
        relevance_scores[idx] = min(score, 100)
    
    return relevance_scores

def extract_domain_from_url(url):
    """Extrait le domaine d'une URL"""
    try:
        from urllib.parse import urlparse
        domain = urlparse(str(url)).netloc
        return domain.replace('www.', '').lower()
    except:
        return str(url)

def detect_file_type_and_process(df):
    """Détecte le type de fichier Ahrefs et le traite en conséquence"""
    
    if 'Referring page URL' in df.columns:
        st.info("📄 Fichier Referring Pages détecté - Conversion en format Domains...")
        
        df['Domain_extracted'] = df['Referring page URL'].apply(extract_domain_from_url)
        
        domain_columns = ['Domain_extracted']
        agg_columns = {}
        
        numeric_cols = ['Domain rating', 'Domain traffic', 'Referring domains', 'Page traffic']
        for col in numeric_cols:
            if col in df.columns:
                agg_columns[col] = 'first'
        
        competitor_cols = [col for col in df.columns if col.startswith('www.')]
        for col in competitor_cols:
            if col in df.columns:
                agg_columns[col] = 'sum'
        
        if 'Intersect' in df.columns:
            agg_columns['Intersect'] = 'first'
        
        domains_df = df.groupby('Domain_extracted').agg(agg_columns).reset_index()
        domains_df = domains_df.rename(columns={'Domain_extracted': 'Domain'})
        
        return domains_df, 'pages_converted'
    
    elif any(col in df.columns for col in ['Domain', 'domain', 'Domaine']):
        return df, 'domains'
    
    else:
        return None, 'unknown'

def calculate_priority_score_vectorized(df, keywords_data=None, pages_data=None):
    """Version vectorisée du calcul de score de priorité"""
    
    domain_column = None
    possible_domain_columns = ['Domain', 'domain', 'Domaine', 'domaine']
    for col in possible_domain_columns:
        if col in df.columns:
            domain_column = col
            break
    
    if domain_column is None:
        st.error("Impossible de trouver la colonne des domaines dans le fichier Ahrefs. Colonnes disponibles: " + ", ".join(df.columns))
        return pd.Series(0, index=df.index), pd.Series(0, index=df.index)
    
    dr_column = None
    traffic_column = None
    
    possible_dr_columns = ['Domain rating', 'domain rating', 'DR', 'dr', 'Domain Rating']
    possible_traffic_columns = ['Domain traffic', 'domain traffic', 'Traffic', 'traffic', 'Domain Traffic']
    
    for col in possible_dr_columns:
        if col in df.columns:
            dr_column = col
            break
    
    for col in possible_traffic_columns:
        if col in df.columns:
            traffic_column = col
            break
    
    dr = pd.to_numeric(df[dr_column], errors='coerce').fillna(0) if dr_column else pd.Series(0, index=df.index)
    traffic = pd.to_numeric(df[traffic_column], errors='coerce').fillna(0) if traffic_column else pd.Series(0, index=df.index)
    
    competitor_columns = [col for col in df.columns if col.startswith('www.') and 'explore-grandest.com' not in col]
    
    if competitor_columns:
        competitor_data = df[competitor_columns].fillna(0)
        competitor_links = (competitor_data > 0).sum(axis=1)
        gap_normalized = (competitor_links / len(competitor_columns)) * 100
    else:
        competitor_links = pd.Series(0, index=df.index)
        gap_normalized = pd.Series(0, index=df.index)
    
    thematic_scores = calculate_thematic_relevance_optimized(df[domain_column], keywords_data, pages_data)
    
    priority_scores = (
        dr * 0.2 +
        np.minimum(traffic / 10000, 100) * 0.2 +
        gap_normalized * 0.3 +
        thematic_scores * 0.3
    )
    
    return priority_scores.round(2), competitor_links

def analyze_serp_data(serp_files):
    """Analyse les fichiers SERP pour générer des benchmarks"""
    serp_analysis_results = {}
    
    for serp_file in serp_files:
        try:
            file_extension = serp_file.name.split('.')[-1].lower()
            
            # Gestion des fichiers Excel
            if file_extension in ['xlsx', 'xls']:
                serp_df = pd.read_excel(serp_file)
            else:
                # Gestion CSV (code existant)
                content = serp_file.read()
                decoded_content = content.decode('utf-16le')
                decoded_content = decoded_content.replace('\x00', '').replace('\ufeff', '')
                serp_df = pd.read_csv(StringIO(decoded_content), sep='\t')
            
            if len(serp_df) == 0:
                continue
                
            keyword = serp_df['Keyword'].iloc[0] if 'Keyword' in serp_df.columns else f"Mot-clé_{len(serp_analysis_results)+1}"
            
            numeric_columns = ['Position', 'Backlinks', 'Referring Domains', 'Domain rating', 'URL rating', 'Traffic']
            for col in numeric_columns:
                if col in serp_df.columns:
                    serp_df[col] = pd.to_numeric(serp_df[col], errors='coerce')
            
            top_10 = serp_df[serp_df['Position'] <= 10].copy()
            
            if len(top_10) == 0:
                continue
            
            analysis = {
                'keyword': keyword,
                'total_results': len(top_10),
                'segments': {}
            }
            
            segments = {
                'top_1': top_10[top_10['Position'] == 1],
                'top_3': top_10[top_10['Position'] <= 3],
                'top_5': top_10[top_10['Position'] <= 5],
                'top_10': top_10
            }
            
            for segment_name, segment_df in segments.items():
                if len(segment_df) > 0:
                    analysis['segments'][segment_name] = {
                        'count': len(segment_df),
                        'backlinks_median': segment_df['Backlinks'].median(),
                        'backlinks_mean': segment_df['Backlinks'].mean(),
                        'rd_median': segment_df['Referring Domains'].median(),
                        'rd_mean': segment_df['Referring Domains'].mean(),
                        'dr_median': segment_df['Domain rating'].median(),
                        'dr_mean': segment_df['Domain rating'].mean(),
                        'ur_median': segment_df['URL rating'].median() if 'URL rating' in segment_df.columns else 0,
                        'ur_mean': segment_df['URL rating'].mean() if 'URL rating' in segment_df.columns else 0,
                        'traffic_median': segment_df['Traffic'].median() if 'Traffic' in segment_df.columns else 0,
                        'traffic_mean': segment_df['Traffic'].mean() if 'Traffic' in segment_df.columns else 0,
                    }
            
            detail_columns = ['Position', 'URL', 'Backlinks', 'Referring Domains', 'Domain rating', 'URL rating', 'Traffic', 'Title']
            available_columns = [col for col in detail_columns if col in top_10.columns]
            analysis['detailed_data'] = top_10[available_columns].to_dict('records')
            
            serp_analysis_results[keyword] = analysis
            
        except Exception as e:
            st.warning(f"Erreur lors de l'analyse du fichier {serp_file.name}: {str(e)}")
            continue
    
    return serp_analysis_results

def generate_serp_recommendations(serp_analysis):
    """Génère des recommandations basées sur l'analyse SERP (utilise la MOYENNE)"""
    recommendations = {}
    
    for keyword, analysis in serp_analysis.items():
        reco = {
            'keyword': keyword,
            'recommendations': []
        }
        
        if 'top_1' in analysis['segments']:
            top1_data = analysis['segments']['top_1']
            reco['recommendations'].append({
                'position_target': 'Position #1',
                'backlinks_target': int(top1_data['backlinks_mean']),
                'rd_target': int(top1_data['rd_mean']),
                'dr_target': int(top1_data['dr_mean']),
                'ur_target': int(top1_data['ur_mean']),
                'message': f"Pour viser la 1ère position sur '{keyword}': {int(top1_data['backlinks_mean'])} backlinks, {int(top1_data['rd_mean'])} RD, DR {int(top1_data['dr_mean'])} (moyennes)"
            })
        
        if 'top_3' in analysis['segments']:
            top3_data = analysis['segments']['top_3']
            reco['recommendations'].append({
                'position_target': 'Top 3',
                'backlinks_target': int(top3_data['backlinks_mean']),
                'rd_target': int(top3_data['rd_mean']),
                'dr_target': int(top3_data['dr_mean']),
                'ur_target': int(top3_data['ur_mean']),
                'message': f"Pour être dans le top 3 sur '{keyword}': {int(top3_data['backlinks_mean'])} backlinks, {int(top3_data['rd_mean'])} RD, DR {int(top3_data['dr_mean'])} (moyennes)"
            })
        
        if 'top_5' in analysis['segments']:
            top5_data = analysis['segments']['top_5']
            reco['recommendations'].append({
                'position_target': 'Top 5',
                'backlinks_target': int(top5_data['backlinks_mean']),
                'rd_target': int(top5_data['rd_mean']),
                'dr_target': int(top5_data['dr_mean']),
                'ur_target': int(top5_data['ur_mean']),
                'message': f"Pour être dans le top 5 sur '{keyword}': {int(top5_data['backlinks_mean'])} backlinks, {int(top5_data['rd_mean'])} RD, DR {int(top5_data['dr_mean'])} (moyennes)"
            })
        
        recommendations[keyword] = reco
    
    return recommendations

# Interface utilisateur
st.sidebar.header("📁 Upload des fichiers")

ahrefs_domains_file = st.sidebar.file_uploader(
    "Export Ahrefs - Referring Domains",
    type=['csv', 'xlsx', 'xls'],
    help="Export CSV ou Excel des domaines référents depuis Ahrefs"
)

ahrefs_pages_file = st.sidebar.file_uploader(
    "Export Ahrefs - Referring Pages",
    type=['csv', 'xlsx', 'xls'],
    help="Export CSV ou Excel des pages référentes depuis Ahrefs"
)

gsc_keywords_file = st.sidebar.file_uploader(
    "Export GSC - Requêtes",
    type=['csv', 'xlsx', 'xls'],
    help="Export CSV ou Excel des requêtes depuis Google Search Console"
)

gsc_pages_file = st.sidebar.file_uploader(
    "Export GSC - Pages",
    type=['csv', 'xlsx', 'xls'],
    help="Export CSV ou Excel des pages depuis Google Search Console"
)

strategic_keywords_file = st.sidebar.file_uploader(
    "Mots-clés stratégiques",
    type=['xlsx', 'csv', 'xls'],
    help="Fichier Excel ou CSV contenant vos mots-clés stratégiques"
)

serp_analysis_files = st.sidebar.file_uploader(
    "Analyse SERPs (Optionnel)",
    type=['csv', 'xlsx', 'xls'],
    accept_multiple_files=True,
    help="Fichiers CSV ou Excel d'analyse SERP Ahrefs (10 maximum) - Un fichier par mot-clé"
)

st.sidebar.header("🎛️ Paramètres de filtrage")

if ahrefs_domains_file is not None:
    
    with st.spinner("Chargement des données Ahrefs Domains..."):
        ahrefs_domains_df = read_file_universal(ahrefs_domains_file)
    
    if ahrefs_domains_df is not None:
        processed_df, file_type = detect_file_type_and_process(ahrefs_domains_df)
        
        if processed_df is None:
            st.error("❌ Type de fichier non reconnu. Veuillez uploader un fichier Ahrefs 'Referring Domains' ou 'Referring Pages'.")
            st.stop()
        
        if file_type == 'pages_converted':
            st.warning("⚠️ Fichier Referring Pages détecté et converti automatiquement en format Domains. Pour de meilleurs résultats, utilisez directement l'export 'Referring Domains' d'Ahrefs.")
        
        ahrefs_domains_df = processed_df
        st.success(f"✅ Fichier Ahrefs traité : {len(ahrefs_domains_df)} domaines uniques")
        
        competitor_columns = [col for col in ahrefs_domains_df.columns if col.startswith('www.')]
        main_site = 'www.explore-grandest.com/' if 'www.explore-grandest.com/' in competitor_columns else competitor_columns[0] if competitor_columns else None
        other_competitors = [col for col in competitor_columns if col != main_site] if main_site else competitor_columns
        
        if main_site:
            st.sidebar.write(f"**Site principal détecté :** {main_site}")
        st.sidebar.write(f"**Concurrents détectés :** {len(other_competitors)}")
        
        if len(other_competitors) == 0:
            st.error("❌ Aucun concurrent détecté dans le fichier. Vérifiez que votre export contient les colonnes des sites concurrents.")
            st.stop()
        
        max_competitors = len(other_competitors)
        min_competitors_filter = st.sidebar.slider(
            "Minimum de concurrents ayant des liens",
            min_value=1,
            max_value=max_competitors,
            value=1,
            help="Filtrer les domaines qui font des liens vers au moins X concurrents"
        )
        
        st.sidebar.subheader("Filtres sur les métriques")
        
        dr_range = st.sidebar.slider(
            "Fourchette Domain Rating",
            min_value=0,
            max_value=100,
            value=(20, 100),
            help="Filtrer par fourchette de Domain Rating"
        )
        min_dr, max_dr = dr_range
        
        traffic_range = st.sidebar.slider(
            "Fourchette de trafic",
            min_value=0,
            max_value=10000000,
            value=(1000, 10000000),
            step=1000,
            help="Filtrer par fourchette de trafic mensuel"
        )
        min_traffic, max_traffic = traffic_range
        
        keywords_data = None
        pages_data = None
        serp_analysis_data = None
        
        if strategic_keywords_file is not None:
            if strategic_keywords_file.name.endswith(('.xlsx', '.xls')):
                keywords_data = pd.read_excel(strategic_keywords_file)
            else:
                keywords_data = pd.read_csv(strategic_keywords_file)
            st.success(f"✅ Mots-clés stratégiques chargés : {len(keywords_data)} mots-clés")
        
        if gsc_pages_file is not None:
            if gsc_pages_file.name.endswith(('.xlsx', '.xls')):
                pages_data = pd.read_excel(gsc_pages_file)
            else:
                pages_data = pd.read_csv(gsc_pages_file)
            if 'CTR' in pages_data.columns:
                pages_data['CTR'] = pages_data['CTR'].apply(clean_percentage)
            st.success(f"✅ Pages GSC chargées : {len(pages_data)} pages")
        
        if gsc_keywords_file is not None:
            if gsc_keywords_file.name.endswith(('.xlsx', '.xls')):
                gsc_keywords_data = pd.read_excel(gsc_keywords_file)
            else:
                gsc_keywords_data = pd.read_csv(gsc_keywords_file)
            if 'CTR' in gsc_keywords_data.columns:
                gsc_keywords_data['CTR'] = gsc_keywords_data['CTR'].apply(clean_percentage)
            st.success(f"✅ Requêtes GSC chargées : {len(gsc_keywords_data)} requêtes")
        
        if serp_analysis_files and len(serp_analysis_files) > 0:
            with st.spinner("Analyse des fichiers SERP..."):
                serp_analysis_data = analyze_serp_data(serp_analysis_files)
                if serp_analysis_data:
                    st.success(f"✅ Analyse SERP : {len(serp_analysis_data)} mots-clés analysés")
                else:
                    st.warning("Aucune donnée SERP valide trouvée")
        
        with st.spinner("Application des filtres et calcul des scores..."):
            filtered_df = ahrefs_domains_df.copy()
            
            dr_column = None
            traffic_column = None
            domain_column = None
            
            possible_dr_columns = ['Domain rating', 'domain rating', 'DR', 'dr', 'Domain Rating']
            possible_traffic_columns = ['Domain traffic', 'domain traffic', 'Traffic', 'traffic', 'Domain Traffic']
            possible_domain_columns = ['Domain', 'domain', 'Domaine', 'domaine']
            
            for col in possible_dr_columns:
                if col in filtered_df.columns:
                    dr_column = col
                    break
            
            for col in possible_traffic_columns:
                if col in filtered_df.columns:
                    traffic_column = col
                    break
            
            for col in possible_domain_columns:
                if col in filtered_df.columns:
                    domain_column = col
                    break
            
            if not domain_column:
                st.error("❌ Impossible de trouver la colonne des domaines. Colonnes disponibles: " + ", ".join(filtered_df.columns[:10]))
                st.stop()
            
            if dr_column:
                filtered_df['Domain rating'] = pd.to_numeric(filtered_df[dr_column], errors='coerce').fillna(0)
            else:
                filtered_df['Domain rating'] = 0
                st.warning("⚠️ Colonne Domain Rating non trouvée")
            
            if traffic_column:
                filtered_df['Domain traffic'] = pd.to_numeric(filtered_df[traffic_column], errors='coerce').fillna(0)
            else:
                filtered_df['Domain traffic'] = 0
                st.warning("⚠️ Colonne Domain Traffic non trouvée")
            
            if domain_column != 'Domain':
                filtered_df['Domain'] = filtered_df[domain_column]
            
            mask = (
                (filtered_df['Domain rating'] >= min_dr) &
                (filtered_df['Domain rating'] <= max_dr) &
                (filtered_df['Domain traffic'] >= min_traffic) &
                (filtered_df['Domain traffic'] <= max_traffic)
            )
            filtered_df = filtered_df[mask]
            
            competitor_data = filtered_df[other_competitors].fillna(0)
            filtered_df['competitor_links_count'] = (competitor_data > 0).sum(axis=1)
            
            filtered_df = filtered_df[filtered_df['competitor_links_count'] >= min_competitors_filter]
            
            if len(filtered_df) > 0:
                priority_scores, competitor_links = calculate_priority_score_vectorized(
                    filtered_df, keywords_data, pages_data
                )
                filtered_df['priority_score'] = priority_scores
                
                filtered_df = filtered_df.sort_values('priority_score', ascending=False)
                
                filtered_df['gap_opportunity'] = filtered_df['competitor_links_count']
                filtered_df['traffic_potential'] = filtered_df['Domain traffic'] * filtered_df['Domain rating'] / 100
        
        filtered_pages_df = None
        if ahrefs_pages_file is not None:
            with st.spinner("Traitement des referring pages..."):
                ahrefs_pages_df = read_file_universal(ahrefs_pages_file)
                if ahrefs_pages_df is not None:
                    ahrefs_pages_df['Domain rating'] = pd.to_numeric(ahrefs_pages_df.get('Domain rating', 0), errors='coerce').fillna(0)
                    ahrefs_pages_df['Domain traffic'] = pd.to_numeric(ahrefs_pages_df.get('Domain traffic', 0), errors='coerce').fillna(0)
                    ahrefs_pages_df['Page traffic'] = pd.to_numeric(ahrefs_pages_df.get('Page traffic', 0), errors='coerce').fillna(0)
                    ahrefs_pages_df['UR'] = pd.to_numeric(ahrefs_pages_df.get('UR', 0), errors='coerce').fillna(0)
                    
                    priority_domains = set(filtered_df['Domain'].tolist())
                    if 'Referring page URL' in ahrefs_pages_df.columns:
                        ahrefs_pages_df['extracted_domain'] = ahrefs_pages_df['Referring page URL'].apply(
                            lambda x: urlparse(str(x)).netloc.replace('www.', '') if pd.notna(x) else ''
                        )
                        filtered_pages_df = ahrefs_pages_df[ahrefs_pages_df['extracted_domain'].isin(priority_domains)].copy()
                    elif 'Domain' in ahrefs_pages_df.columns:
                        filtered_pages_df = ahrefs_pages_df[ahrefs_pages_df['Domain'].isin(priority_domains)].copy()
                    
                    if filtered_pages_df is not None and len(filtered_pages_df) > 0:
                        filtered_pages_df['page_score'] = (
                            filtered_pages_df['Domain rating'] * 0.3 +
                            np.minimum(filtered_pages_df['Page traffic'] / 1000, 100) * 0.3 +
                            filtered_pages_df['UR'] * 0.4
                        )
                        filtered_pages_df = filtered_pages_df.sort_values('page_score', ascending=False)
        
        st.header("📊 Résultats de l'analyse")
        
        tab_names = ["📈 Tableau de bord", "🎯 Referring Domains"]
        
        if filtered_pages_df is not None and len(filtered_pages_df) > 0:
            tab_names.append("📄 Referring Pages")
        
        if serp_analysis_data:
            tab_names.append("🎯 Analyse SERPs")
        
        tab_names.extend(["📁 Fichiers d'entrée", "💾 Export CSV"])
        
        tabs = st.tabs(tab_names)
        
        tab_index = 0
        
        with tabs[tab_index]:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Domaines analysés",
                    len(ahrefs_domains_df),
                    delta=f"+{len(filtered_df)} après filtrage"
                )
            
            with col2:
                avg_dr = filtered_df['Domain rating'].mean()
                st.metric(
                    "DR moyen",
                    f"{avg_dr:.1f}",
                    delta=f"Fourchette: {min_dr}-{max_dr}"
                )
            
            with col3:
                avg_gap = filtered_df['competitor_links_count'].mean()
                st.metric(
                    "Gap moyen",
                    f"{avg_gap:.1f}",
                    delta=f"concurrents/domaine"
                )
            
            with col4:
                total_traffic = filtered_df['Domain traffic'].sum()
                st.metric(
                    "Trafic total potentiel",
                    f"{total_traffic/1000000:.1f}M",
                    delta="visiteurs/mois"
                )
            
            st.subheader("📈 Visualisations")
            
            fig_scatter = px.scatter(
                filtered_df.head(50),
                x='Domain rating',
                y='Domain traffic',
                size='priority_score',
                color='competitor_links_count',
                hover_data=['Domain', 'priority_score', 'gap_opportunity'],
                title="Top 50 - Opportunités de Netlinking",
                labels={
                    'Domain rating': 'Domain Rating',
                    'Domain traffic': 'Trafic du Domaine',
                    'competitor_links_count': 'Nombre de concurrents'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            top_domains = filtered_df.head(20)
            fig_bar = px.bar(
                top_domains,
                x='priority_score',
                y='Domain',
                orientation='h',
                title="Top 20 - Domaines par Score de Priorité",
                labels={'priority_score': 'Score de Priorité', 'Domain': 'Domaine'}
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.subheader("🎯 Analyse par segments")
            
            def categorize_priority(score):
                if score >= 70:
                    return "🔥 Priorité Maximale"
                elif score >= 50:
                    return "⚡ Priorité Élevée"
                elif score >= 30:
                    return "🎯 Priorité Moyenne"
                else:
                    return "📝 Priorité Faible"
            
            filtered_df['priority_category'] = filtered_df['priority_score'].apply(categorize_priority)
            
            priority_counts = filtered_df['priority_category'].value_counts()
            fig_pie = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Répartition des opportunités par niveau de priorité"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.subheader("💡 Recommandations automatiques")
            
            top_3 = filtered_df.head(3)
            
            st.write("**🏆 Top 3 des domaines à contacter en priorité :**")
            
            for i, (_, domain) in enumerate(top_3.iterrows(), 1):
                with st.expander(f"#{i} - {domain['Domain']} (Score: {domain['priority_score']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Domain Rating", f"{domain['Domain rating']}")
                        st.metric("Trafic mensuel", f"{domain['Domain traffic']:,.0f}")
                    
                    with col2:
                        st.metric("Concurrents liés", f"{domain['competitor_links_count']}")
                        st.metric("Potentiel trafic", f"{domain['traffic_potential']:,.0f}")
                    
                    linked_competitors = []
                    for comp in other_competitors:
                        if pd.notna(domain[comp]) and int(domain[comp]) > 0:
                            linked_competitors.append(f"{comp} ({int(domain[comp])} liens)")
                    
                    if linked_competitors:
                        st.write("**Concurrents ayant des liens :**")
                        st.write(" • ".join(linked_competitors))
        
        tab_index += 1
        with tabs[tab_index]:
            st.subheader("📋 Tableau détaillé des opportunités - Referring Domains")
            
            display_columns = [
                'Domain', 'Domain rating', 'Domain traffic', 'priority_score',
                'competitor_links_count', 'gap_opportunity'
            ] + other_competitors[:3]
            
            display_df = filtered_df[display_columns].head(100)
            
            display_df = display_df.round(2)
            display_df.columns = [
                'Domaine', 'DR', 'Trafic', 'Score Priorité',
                'Nb Concurrents', 'Opportunité Gap'
            ] + [f'Concurrent {i+1}' for i in range(len(other_competitors[:3]))]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                height=600,
                column_config={
                    "Score Priorité": st.column_config.ProgressColumn(
                        "Score Priorité",
                        help="Score calculé sur 100",
                        min_value=0,
                        max_value=100,
                    ),
                    "DR": st.column_config.NumberColumn(
                        "DR",
                        help="Domain Rating Ahrefs",
                        min_value=0,
                        max_value=100,
                        format="%d",
                    ),
                    "Trafic": st.column_config.NumberColumn(
                        "Trafic",
                        help="Trafic mensuel estimé",
                        format="%d",
                    ),
                }
            )
        
        if filtered_pages_df is not None and len(filtered_pages_df) > 0:
            tab_index += 1
            with tabs[tab_index]:
                st.subheader("📄 Pages référentes prioritaires à cibler")
                
                page_display_columns = []
                available_columns = filtered_pages_df.columns.tolist()
                
                essential_cols = ['Referring page title', 'Referring page URL', 'Domain', 'Domain rating', 'UR', 'Page traffic', 'page_score']
                for col in essential_cols:
                    if col in available_columns:
                        page_display_columns.append(col)
                
                for comp in other_competitors[:2]:
                    if comp in available_columns:
                        page_display_columns.append(comp)
                
                pages_display_df = filtered_pages_df[page_display_columns].head(200)
                
                rename_dict = {
                    'Referring page title': 'Titre de la page',
                    'Referring page URL': 'URL de la page',
                    'Domain': 'Domaine',
                    'Domain rating': 'DR',
                    'UR': 'UR',
                    'Page traffic': 'Trafic page',
                    'page_score': 'Score page'
                }
                
                pages_display_df = pages_display_df.rename(columns=rename_dict)
                pages_display_df = pages_display_df.round(2)
                
                st.dataframe(
                    pages_display_df,
                    use_container_width=True,
                    height=600,
                    column_config={
                        "Score page": st.column_config.ProgressColumn(
                            "Score page",
                            help="Score calculé pour la page",
                            min_value=0,
                            max_value=100,
                        ),
                        "DR": st.column_config.NumberColumn(
                            "DR",
                            help="Domain Rating",
                            min_value=0,
                            max_value=100,
                            format="%d",
                        ),
                        "UR": st.column_config.NumberColumn(
                            "UR",
                            help="URL Rating",
                            min_value=0,
                            max_value=100,
                            format="%d",
                        ),
                        "URL de la page": st.column_config.LinkColumn("URL de la page"),
                    }
                )
                
                st.write("**📊 Statistiques des pages référentes :**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Pages analysées", len(filtered_pages_df))
                
                with col2:
                    avg_ur = filtered_pages_df['UR'].mean() if 'UR' in filtered_pages_df.columns else 0
                    st.metric("UR moyen", f"{avg_ur:.1f}")
                
                with col3:
                    avg_page_traffic = filtered_pages_df['Page traffic'].mean() if 'Page traffic' in filtered_pages_df.columns else 0
                    st.metric("Trafic page moyen", f"{avg_page_traffic:.0f}")
        
        if serp_analysis_data:
            tab_index += 1
            with tabs[tab_index]:
                st.subheader("🎯 Analyse des SERPs - Benchmarks par mot-clé")
                
                selected_keyword = st.selectbox(
                    "Choisissez un mot-clé à analyser",
                    options=list(serp_analysis_data.keys()),
                    key="serp_keyword_selector"
                )
                
                if selected_keyword:
                    analysis = serp_analysis_data[selected_keyword]
                    
                    st.write(f"**Analyse pour le mot-clé : {selected_keyword}**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Résultats analysés", analysis['total_results'])
                    
                    if 'top_1' in analysis['segments']:
                        with col2:
                            top1_bl = analysis['segments']['top_1']['backlinks_mean']
                            st.metric("Backlinks #1", f"{int(top1_bl)}")
                        
                        with col3:
                            top1_dr = analysis['segments']['top_1']['dr_mean']
                            st.metric("DR #1", f"{int(top1_dr)}")
                        
                        with col4:
                            top1_rd = analysis['segments']['top_1']['rd_mean']
                            st.metric("RD #1", f"{int(top1_rd)}")
                    
                    st.subheader("📊 Benchmarks par position")
                    
                    benchmark_data = []
                    for segment_name, segment_data in analysis['segments'].items():
                        if segment_name == 'top_1':
                            display_name = "Position #1"
                        elif segment_name == 'top_3':
                            display_name = "Top 3"
                        elif segment_name == 'top_5':
                            display_name = "Top 5"
                        elif segment_name == 'top_10':
                            display_name = "Top 10"
                        else:
                            display_name = segment_name
                        
                        benchmark_data.append({
                            'Segment': display_name,
                            'Nb sites': segment_data['count'],
                            'Backlinks (moyen)': int(segment_data['backlinks_mean']),
                            'RD (moyen)': int(segment_data['rd_mean']),
                            'DR (moyen)': int(segment_data['dr_mean']),
                            'UR (moyen)': int(segment_data['ur_mean']),
                            'Trafic (moyen)': int(segment_data['traffic_mean']),
                            'Backlinks (médian)': int(segment_data['backlinks_median']),
                            'RD (médian)': int(segment_data['rd_median']),
                            'DR (médian)': int(segment_data['dr_median'])
                        })
                    
                    benchmark_df = pd.DataFrame(benchmark_data)
                    st.dataframe(benchmark_df, use_container_width=True)
                    
                    st.subheader("📈 Visualisation des benchmarks")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_bl = px.bar(
                            benchmark_df,
                            x='Segment',
                            y='Backlinks (moyen)',
                            title="Backlinks moyens par segment",
                            text='Backlinks (moyen)'
                        )
                        fig_bl.update_traces(texttemplate='%{text}', textposition='outside')
                        st.plotly_chart(fig_bl, use_container_width=True)
                    
                    with col2:
                        fig_dr = px.bar(
                            benchmark_df,
                            x='Segment',
                            y='DR (moyen)',
                            title="Domain Rating moyen par segment",
                            text='DR (moyen)',
                            color='DR (moyen)',
                            color_continuous_scale='Viridis'
                        )
                        fig_dr.update_traces(texttemplate='%{text}', textposition='outside')
                        st.plotly_chart(fig_dr, use_container_width=True)
                    
                    st.subheader("💡 Recommandations de netlinking")
                    
                    recommendations = generate_serp_recommendations({selected_keyword: analysis})
                    
                    if selected_keyword in recommendations:
                        reco_data = recommendations[selected_keyword]
                        
                        for reco in reco_data['recommendations']:
                            with st.expander(f"🎯 {reco['position_target']}", expanded=True):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Backlinks cible", reco['backlinks_target'])
                                
                                with col2:
                                    st.metric("RD cible", reco['rd_target'])
                                
                                with col3:
                                    st.metric("DR cible", reco['dr_target'])
                                
                                st.info(reco['message'])
                    
                    st.subheader("🔍 Détail du top 10")
                    
                    detailed_df = pd.DataFrame(analysis['detailed_data'])
                    if not detailed_df.empty:
                        display_detailed = detailed_df.copy()
                        numeric_cols = ['Position', 'Backlinks', 'Referring Domains', 'Domain rating', 'URL rating', 'Traffic']
                        for col in numeric_cols:
                            if col in display_detailed.columns:
                                display_detailed[col] = pd.to_numeric(display_detailed[col], errors='coerce').fillna(0).astype(int)
                        
                        st.dataframe(
                            display_detailed,
                            use_container_width=True,
                            height=400,
                            column_config={
                                "Position": st.column_config.NumberColumn(
                                    "Pos",
                                    format="%d",
                                ),
                                "URL": st.column_config.LinkColumn("URL"),
                                "Backlinks": st.column_config.NumberColumn(
                                    "Backlinks",
                                    format="%d",
                                ),
                                "Referring Domains": st.column_config.NumberColumn(
                                    "RD",
                                    format="%d",
                                ),
                                "Domain rating": st.column_config.NumberColumn(
                                    "DR",
                                    format="%d",
                                ),
                                "URL rating": st.column_config.NumberColumn(
                                    "UR",
                                    format="%d",
                                ),
                                "Traffic": st.column_config.NumberColumn(
                                    "Trafic",
                                    format="%d",
                                ),
                            }
                        )
        
        tab_index += 1
        with tabs[tab_index]:
            st.subheader("📁 Fichiers d'entrée - Aperçu des données")
            
            sub_tab_names = []
            if keywords_data is not None:
                sub_tab_names.append("🎯 Mots-clés stratégiques")
            if 'gsc_keywords_data' in locals() and gsc_keywords_data is not None:
                sub_tab_names.append("📊 GSC Requêtes")
            if pages_data is not None:
                sub_tab_names.append("📄 GSC Pages")
            if serp_analysis_data:
                sub_tab_names.append("🎯 Analyse SERPs")
            
            if sub_tab_names:
                sub_tabs = st.tabs(sub_tab_names)
                
                sub_tab_idx = 0
                
                if keywords_data is not None:
                    with sub_tabs[sub_tab_idx]:
                        st.write(f"**{len(keywords_data)} mots-clés stratégiques chargés**")
                        st.dataframe(keywords_data.head(20), use_container_width=True)
                    sub_tab_idx += 1
                
                if 'gsc_keywords_data' in locals() and gsc_keywords_data is not None:
                    with sub_tabs[sub_tab_idx]:
                        st.write(f"**{len(gsc_keywords_data)} requêtes GSC chargées**")
                        st.dataframe(gsc_keywords_data.head(20), use_container_width=True)
                    sub_tab_idx += 1
                
                if pages_data is not None:
                    with sub_tabs[sub_tab_idx]:
                        st.write(f"**{len(pages_data)} pages GSC chargées**")
                        st.dataframe(pages_data.head(20), use_container_width=True)
                    sub_tab_idx += 1
                
                if serp_analysis_data:
                    with sub_tabs[sub_tab_idx]:
                        st.write(f"**{len(serp_analysis_data)} mots-clés analysés dans les SERPs**")
                        summary_data = []
                        for keyword, analysis in serp_analysis_data.items():
                            summary_data.append({
                                'Mot-clé': keyword,
                                'Résultats': analysis['total_results'],
                                'DR #1': int(analysis['segments']['top_1']['dr_mean']) if 'top_1' in analysis['segments'] else 'N/A',
                                'Backlinks #1': int(analysis['segments']['top_1']['backlinks_mean']) if 'top_1' in analysis['segments'] else 'N/A'
                            })
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
            else:
                st.info("Aucun fichier optionnel chargé. Uploadez vos fichiers GSC et mots-clés stratégiques pour enrichir l'analyse.")
        
        tab_index += 1
        with tabs[tab_index]:
            st.subheader("💾 Télécharger les résultats")
            
            if len(filtered_df) > 0:
                # Préparer tous les DataFrames pour l'export Excel
                export_df = filtered_df.copy()
                
                export_columns = {
                    'Domain': 'Domaine',
                    'Domain rating': 'Domain_Rating',
                    'Domain traffic': 'Trafic_Mensuel',
                    'priority_score': 'Score_Priorite',
                    'competitor_links_count': 'Nb_Concurrents_Lies',
                    'gap_opportunity': 'Opportunite_Gap',
                    'traffic_potential': 'Potentiel_Trafic'
                }
                
                for i, comp in enumerate(other_competitors):
                    clean_name = comp.replace('www.', '').replace('/', '').replace('.com', '').replace('.fr', '')
                    export_columns[comp] = f'{clean_name}_Liens'
                
                export_df = export_df.rename(columns=export_columns)
                
                key_columns = [
                    'Domaine', 'Domain_Rating', 'Trafic_Mensuel', 'Score_Priorite',
                    'Nb_Concurrents_Lies', 'Opportunite_Gap', 'Potentiel_Trafic'
                ]
                
                competitor_columns_renamed = []
                for comp in other_competitors:
                    clean_name = comp.replace('www.', '').replace('/', '').replace('.com', '').replace('.fr', '')
                    competitor_columns_renamed.append(f'{clean_name}_Liens')
                
                final_columns = key_columns + competitor_columns_renamed
                final_export_df = export_df[final_columns].round(2)
                
                # Préparer le fichier Excel avec plusieurs onglets
                from io import BytesIO
                
                excel_buffer = BytesIO()
                
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    # Onglet 1: Tous les domaines
                    final_export_df.to_excel(writer, sheet_name='Domains_Complet', index=False)
                    
                    # Onglet 2: Top 50 domaines
                    final_export_df.head(50).to_excel(writer, sheet_name='Top_50_Domains', index=False)
                    
                    # Onglet 3: Domaines priorité maximale (score >= 70)
                    high_priority = final_export_df[final_export_df['Score_Priorite'] >= 70]
                    if len(high_priority) > 0:
                        high_priority.to_excel(writer, sheet_name='Priorite_Maximale', index=False)
                    
                    # Onglet 4: Gaps importants
                    high_gap = final_export_df[final_export_df['Nb_Concurrents_Lies'] >= max_competitors - 1]
                    if len(high_gap) > 0:
                        high_gap.to_excel(writer, sheet_name='Gaps_Importants', index=False)
                    
                    # Onglet 5: DR élevé (>= 70)
                    high_dr = final_export_df[final_export_df['Domain_Rating'] >= 70]
                    if len(high_dr) > 0:
                        high_dr.to_excel(writer, sheet_name='DR_Eleve', index=False)
                    
                    # Onglet 6: Referring Pages (si disponible)
                    if filtered_pages_df is not None and len(filtered_pages_df) > 0:
                        pages_export_df = filtered_pages_df.copy()
                        pages_export_columns = {}
                        
                        if 'Referring page title' in pages_export_df.columns:
                            pages_export_columns['Referring page title'] = 'Titre_Page'
                        if 'Referring page URL' in pages_export_df.columns:
                            pages_export_columns['Referring page URL'] = 'URL_Page'
                        if 'Domain' in pages_export_df.columns:
                            pages_export_columns['Domain'] = 'Domaine'
                        if 'Domain rating' in pages_export_df.columns:
                            pages_export_columns['Domain rating'] = 'Domain_Rating'
                        if 'UR' in pages_export_df.columns:
                            pages_export_columns['UR'] = 'URL_Rating'
                        if 'Page traffic' in pages_export_df.columns:
                            pages_export_columns['Page traffic'] = 'Trafic_Page'
                        if 'page_score' in pages_export_df.columns:
                            pages_export_columns['page_score'] = 'Score_Page'
                        
                        pages_export_df = pages_export_df.rename(columns=pages_export_columns)
                        
                        # Toutes les pages
                        pages_export_df.to_excel(writer, sheet_name='Pages_Completes', index=False)
                        
                        # Top 100 pages
                        pages_export_df.head(100).to_excel(writer, sheet_name='Top_100_Pages', index=False)
                    
                    # Onglets 7-8: Analyse SERP (si disponible)
                    if serp_analysis_data:
                        serp_export_data = []
                        recommendations_export_data = []
                        
                        for keyword, analysis in serp_analysis_data.items():
                            for segment_name, segment_data in analysis['segments'].items():
                                serp_export_data.append({
                                    'Mot_Cle': keyword,
                                    'Segment': segment_name,
                                    'Nombre_Sites': segment_data['count'],
                                    'Backlinks_Moyen': int(segment_data['backlinks_mean']),
                                    'RD_Moyen': int(segment_data['rd_mean']),
                                    'DR_Moyen': int(segment_data['dr_mean']),
                                    'UR_Moyen': int(segment_data['ur_mean']),
                                    'Trafic_Moyen': int(segment_data['traffic_mean']),
                                    'Backlinks_Median': int(segment_data['backlinks_median']),
                                    'RD_Median': int(segment_data['rd_median']),
                                    'DR_Median': int(segment_data['dr_median']),
                                    'UR_Median': int(segment_data['ur_median']),
                                    'Trafic_Median': int(segment_data['traffic_median'])
                                })
                            
                            reco_data = generate_serp_recommendations({keyword: analysis})
                            if keyword in reco_data:
                                for reco in reco_data[keyword]['recommendations']:
                                    recommendations_export_data.append({
                                        'Mot_Cle': keyword,
                                        'Position_Cible': reco['position_target'],
                                        'Backlinks_Cible': reco['backlinks_target'],
                                        'RD_Cible': reco['rd_target'],
                                        'DR_Cible': reco['dr_target'],
                                        'UR_Cible': reco['ur_target'],
                                        'Recommandation': reco['message']
                                    })
                        
                        if serp_export_data:
                            serp_benchmarks_df = pd.DataFrame(serp_export_data)
                            serp_benchmarks_df.to_excel(writer, sheet_name='SERP_Benchmarks', index=False)
                        
                        if recommendations_export_data:
                            recommendations_df = pd.DataFrame(recommendations_export_data)
                            recommendations_df.to_excel(writer, sheet_name='SERP_Recommandations', index=False)
                    
                    # Onglet final: Résumé de l'analyse
                    summary_data = {
                        'Métrique': [
                            'Nombre total de domaines analysés',
                            'Domaines après filtrage',
                            'Score de priorité moyen',
                            'Domain Rating moyen',
                            'Trafic total potentiel (millions)',
                            'Nombre de concurrents détectés',
                            'Fourchette DR appliquée',
                            'Fourchette Trafic appliquée'
                        ],
                        'Valeur': [
                            len(ahrefs_domains_df),
                            len(filtered_df),
                            f"{filtered_df['priority_score'].mean():.2f}/100",
                            f"{filtered_df['Domain rating'].mean():.1f}",
                            f"{filtered_df['Domain traffic'].sum()/1000000:.1f}M",
                            len(other_competitors),
                            f"{min_dr}-{max_dr}",
                            f"{min_traffic:,}-{max_traffic:,}"
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Resume_Analyse', index=False)
                
                excel_buffer.seek(0)
                
                # Informations sur l'analyse
                st.write("**📊 Résumé de l'analyse :**")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Domaines analysés", len(filtered_df))
                
                with col2:
                    st.metric("Score moyen", f"{filtered_df['priority_score'].mean():.1f}/100")
                
                with col3:
                    st.metric("DR moyen", f"{filtered_df['Domain rating'].mean():.1f}")
                
                # Bouton de téléchargement unique
                st.download_button(
                    label="📊 Télécharger l'audit complet (Excel - Tous les onglets)",
                    data=excel_buffer.getvalue(),
                    file_name=f"audit_netlinking_complet_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
                
                # Détail des onglets inclus
                st.write("**📋 Onglets inclus dans le fichier Excel :**")
                
                onglets_info = [
                    f"• **Domains_Complet** : {len(final_export_df)} domaines avec tous les détails",
                    f"• **Top_50_Domains** : Top 50 des domaines prioritaires",
                ]
                
                if len(high_priority) > 0:
                    onglets_info.append(f"• **Priorite_Maximale** : {len(high_priority)} domaines avec score ≥ 70")
                
                if len(high_gap) > 0:
                    onglets_info.append(f"• **Gaps_Importants** : {len(high_gap)} domaines liés à {max_competitors-1}+ concurrents")
                
                if len(high_dr) > 0:
                    onglets_info.append(f"• **DR_Eleve** : {len(high_dr)} domaines avec DR ≥ 70")
                
                if filtered_pages_df is not None and len(filtered_pages_df) > 0:
                    onglets_info.extend([
                        f"• **Pages_Completes** : {len(filtered_pages_df)} pages référentes détaillées",
                        f"• **Top_100_Pages** : Top 100 des pages prioritaires"
                    ])
                
                if serp_analysis_data:
                    onglets_info.extend([
                        f"• **SERP_Benchmarks** : Benchmarks pour {len(serp_analysis_data)} mots-clés",
                        f"• **SERP_Recommandations** : Recommandations de netlinking par position"
                    ])
                
                onglets_info.append("• **Resume_Analyse** : Métriques et paramètres de l'analyse")
                
                for info in onglets_info:
                    st.write(info)
                
                st.info("💡 **Conseil :** Ouvrez le fichier dans Excel ou Google Sheets pour naviguer facilement entre les onglets !")
                
            else:
                st.warning("Aucun domaine ne correspond aux critères de filtrage sélectionnés.")

else:
    st.markdown("**Commencez par uploader votre export Ahrefs 'Referring Domains' dans la barre latérale !**")
    
    with st.expander("📋 Comment utiliser cet outil - Étapes à suivre"):
        st.markdown("""
        1. **Exportez vos données depuis Ahrefs :**
           - Allez dans l'outil "Link Intersect"
           - Ajoutez votre site + vos concurrents
           - Exportez les "Referring Domains" et "Referring Pages"
        
        2. **Exportez vos données depuis Google Search Console :**
           - Allez dans "Performances" > "Requêtes"
           - Exportez les données des requêtes et des pages
        
        3. **Préparez vos mots-clés stratégiques :**
           - Format Excel ou CSV avec colonnes : Keyword, Search Volume, Keyword Difficulty
        
        4. **[OPTIONNEL] Ajoutez vos analyses SERPs :**
           - Depuis Ahrefs > Keywords Explorer > [votre mot-clé] > SERP overview
           - Exportez en CSV (un fichier par mot-clé, 10 maximum)
           - Pour une analyse micro des besoins en backlinks par mot-clé
        
        5. **Uploadez tous les fichiers** dans la barre latérale
        
        6. **Configurez les filtres** selon vos besoins
        """)
    
    with st.expander("🎯 Ce que fait l'outil"):
        st.markdown("""
        - **Analyse les gaps concurrentiels** : Identifie les sites qui font des liens vers vos concurrents mais pas vers vous
        - **Calcule un score de priorité** basé sur :
          - Domain Rating (20%)
          - Trafic du domaine (20%)  
          - Gap concurrentiel (30%)
          - Pertinence thématique (30%)
        - **Analyse micro des SERPs** : Benchmarks de backlinks nécessaires par mot-clé et position cible
        - **Fournit des analyses complètes** avec tableaux de bord, graphiques et exports CSV
        """)
    
    with st.expander("📊 Résultats obtenus"):
        st.markdown("""
        - Tableau de bord avec graphiques interactifs
        - Liste des domaines prioritaires à contacter
        - Liste des pages référentes spécifiques à cibler
        - **NOUVEAU** : Analyse détaillée des besoins en backlinks par mot-clé
        - **NOUVEAU** : Recommandations précises pour atteindre chaque position
        - Fichiers CSV structurés pour vos campagnes
        - Aperçu de tous vos fichiers d'entrée
        """)
    
    with st.expander("📁 Structure des fichiers attendus"):
        st.markdown("""
        **Ahrefs - Referring Domains :**
        ```
        Domain | Domain rating | Domain traffic | Intersect | www.monsite.com | www.concurrent1.com | ...
        ```
        
        **Ahrefs - Referring Pages :**
        ```
        Referring page title | Referring page URL | Domain | Domain rating | UR | Page traffic | Intersect | www.monsite.com | ...
        ```
        
        **Ahrefs - SERP Analysis (NOUVEAU) :**
        ```
        Keyword | URL | Position | Backlinks | Referring Domains | Domain rating | URL rating | Traffic | ...
        ```
        
        **GSC - Requêtes :**
        ```
        Requêtes les plus fréquentes | Clics | Impressions | CTR | Position
        ```
        
        **GSC - Pages :**
        ```
        Pages les plus populaires | Clics | Impressions | CTR | Position
        ```
        
        **Mots-clés stratégiques :**
        ```
        Keyword | Search Volume | Keyword Difficulty | CPC | ...
        ```
        
        **📝 Note :** Tous les fichiers peuvent être au format CSV, Excel (.xlsx) ou ancien Excel (.xls)
        """)

st.markdown("---")
st.markdown("**Développé par [JC Espinosa](https://jc-espinosa.com) pour optimiser vos campagnes de netlinking SEO**")
