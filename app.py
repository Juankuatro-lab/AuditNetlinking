import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from urllib.parse import urlparse

# Imports conditionnels pour éviter les erreurs
try:
    import chardet
except ImportError:
    chardet = None
    st.warning("chardet non disponible - détection d'encodage limitée")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    st.error("Plotly non disponible - installez avec: pip install plotly")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="Audit de netlinking - Création d'une roadmap pour satelliser ta stratégie",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Audit de netlinking - Création d'une roadmap pour satelliser ta stratégie")
st.markdown("**Priorisez vos campagnes de netlinking en croisant les données Ahrefs et GSC**")

# Fonctions utilitaires optimisées
def detect_encoding(file_content):
    """Détecte l'encodage d'un fichier"""
    detected = chardet.detect(file_content)
    return detected['encoding'] if detected['confidence'] > 0.7 else 'utf-8'

def read_ahrefs_csv(uploaded_file):
    """Lit un fichier CSV Ahrefs avec gestion des encodages"""
    try:
        # Lire le contenu brut
        content = uploaded_file.read()
        
        # Essayer UTF-16 LE (format Ahrefs typique)
        try:
            decoded_content = content.decode('utf-16le')
            # Nettoyer les caractères nuls
            decoded_content = decoded_content.replace('\x00', '')
            # Supprimer BOM
            decoded_content = decoded_content.replace('\ufeff', '')
            df = pd.read_csv(StringIO(decoded_content), sep='\t')
            return df
        except:
            pass
        
        # Essayer UTF-8
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='utf-8', sep='\t')
            return df
        except:
            pass
        
        # Essayer détection automatique
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
    
    # Créer un dictionnaire de mots-clés pour la recherche rapide
    keyword_dict = {}
    if keywords_data is not None:
        for _, row in keywords_data.iterrows():
            keyword = str(row.get('Keyword', '')).lower()
            words = set(re.findall(r'\w+', keyword))
            if words:
                keyword_dict[frozenset(words)] = row.get('Search Volume', 0)
    
    # Créer un dictionnaire des pages pour la recherche rapide
    pages_dict = {}
    if pages_data is not None:
        for _, row in pages_data.iterrows():
            page_url = str(row.get('Pages les plus populaires', '')).lower()
            words = set(re.findall(r'\w+', page_url))
            if words:
                pages_dict[frozenset(words)] = row.get('Clics', 0)
    
    # Calculer les scores de pertinence
    for idx, domain in domains_series.items():
        domain_words = set(re.findall(r'\w+', str(domain).lower()))
        score = 0
        
        # Comparer avec les mots-clés
        for keyword_words, volume in keyword_dict.items():
            common_words = domain_words & keyword_words
            if common_words:
                score += len(common_words) * volume / 1000
        
        # Comparer avec les pages
        for page_words, clics in pages_dict.items():
            common_words = domain_words & page_words
            if common_words:
                score += len(common_words) * clics / 100
        
        relevance_scores[idx] = min(score, 100)
    
    return relevance_scores

def analyze_serp_data(serp_files):
    """Analyse les fichiers SERPs pour extraire les benchmarks"""
    all_serp_data = []
    
    for serp_file in serp_files:
        try:
            # Lire le fichier SERP avec la même méthode que les autres fichiers Ahrefs
            serp_df = read_ahrefs_csv(serp_file)
            if serp_df is not None:
                all_serp_data.append(serp_df)
        except Exception as e:
            st.warning(f"Erreur lors de la lecture du fichier SERP {serp_file.name}: {str(e)}")
    
    if not all_serp_data:
        return None
    
    # Combiner tous les DataFrames SERP
    combined_serp = pd.concat(all_serp_data, ignore_index=True)
    
    # Nettoyer les données
    numeric_columns = ['Position', 'Backlinks', 'Referring Domains', 'Domain rating', 'URL rating', 'Traffic']
    for col in numeric_columns:
        if col in combined_serp.columns:
            combined_serp[col] = pd.to_numeric(combined_serp[col], errors='coerce')
    
    # Filtrer les positions valides (1-10 pour le top 10)
    combined_serp = combined_serp[(combined_serp['Position'] >= 1) & (combined_serp['Position'] <= 10)]
    
    return combined_serp

def calculate_serp_benchmarks(serp_df):
    """Calcule les benchmarks par position et par mot-clé"""
    if serp_df is None or len(serp_df) == 0:
        return None
    
    benchmarks = {}
    
    # Analyse par mot-clé
    for keyword in serp_df['Keyword'].unique():
        keyword_data = serp_df[serp_df['Keyword'] == keyword]
        
        if len(keyword_data) == 0:
            continue
        
        keyword_benchmarks = {
            'keyword': keyword,
            'total_results': len(keyword_data),
            'positions': {}
        }
        
        # Benchmarks par groupes de positions
        position_groups = {
            'top_1': keyword_data[keyword_data['Position'] == 1],
            'top_3': keyword_data[keyword_data['Position'] <= 3],
            'top_5': keyword_data[keyword_data['Position'] <= 5],
            'top_10': keyword_data[keyword_data['Position'] <= 10]
        }
        
        for group_name, group_data in position_groups.items():
            if len(group_data) > 0:
                keyword_benchmarks['positions'][group_name] = {
                    'backlinks_median': group_data['Backlinks'].median(),
                    'backlinks_mean': group_data['Backlinks'].mean(),
                    'referring_domains_median': group_data['Referring Domains'].median(),
                    'referring_domains_mean': group_data['Referring Domains'].mean(),
                    'domain_rating_median': group_data['Domain rating'].median(),
                    'domain_rating_mean': group_data['Domain rating'].mean(),
                    'url_rating_median': group_data['URL rating'].median(),
                    'url_rating_mean': group_data['URL rating'].mean(),
                    'traffic_median': group_data['Traffic'].median(),
                    'traffic_mean': group_data['Traffic'].mean(),
                    'count': len(group_data)
                }
        
        benchmarks[keyword] = keyword_benchmarks
    
    return benchmarks

def generate_serp_recommendations(benchmarks):
    """Génère des recommandations basées sur les benchmarks"""
    recommendations = {}
    
    for keyword, data in benchmarks.items():
        recs = {
            'keyword': keyword,
            'recommendations': {}
        }
        
        # Recommandations pour différentes positions cibles
        for position_group in ['top_1', 'top_3', 'top_5']:
            if position_group in data['positions']:
                pos_data = data['positions'][position_group]
                
                target_name = {
                    'top_1': 'position #1',
                    'top_3': 'top 3',
                    'top_5': 'top 5'
                }[position_group]
                
                recs['recommendations'][position_group] = {
                    'target': target_name,
                    'backlinks_target': int(pos_data['backlinks_median']),
                    'referring_domains_target': int(pos_data['referring_domains_median']),
                    'domain_rating_target': int(pos_data['domain_rating_median']),
                    'url_rating_target': int(pos_data['url_rating_median']),
                    'description': f"Pour atteindre {target_name} sur '{keyword}' : visez ~{int(pos_data['backlinks_median'])} backlinks, ~{int(pos_data['referring_domains_median'])} domaines référents, et un DR d'au moins {int(pos_data['domain_rating_median'])}"
                }
        
        recommendations[keyword] = recs
    
    return recommendations

def calculate_priority_score_vectorized(df, keywords_data=None, pages_data=None):
    """Version vectorisée du calcul de score de priorité"""
    
    # Métriques de base (vectorisées)
    dr = pd.to_numeric(df['Domain rating'], errors='coerce').fillna(0)
    traffic = pd.to_numeric(df['Domain traffic'], errors='coerce').fillna(0)
    
    # Identifier les colonnes des concurrents
    competitor_columns = [col for col in df.columns if col.startswith('www.') and 'explore-grandest.com' not in col]
    
    # Calcul vectorisé du gap concurrentiel
    competitor_data = df[competitor_columns].fillna(0)
    competitor_links = (competitor_data > 0).sum(axis=1)
    gap_normalized = (competitor_links / len(competitor_columns)) * 100 if competitor_columns else pd.Series(0, index=df.index)
    
    # Calcul de la pertinence thématique (version optimisée et cachée)
    thematic_scores = calculate_thematic_relevance_optimized(df['Domain'], keywords_data, pages_data)
    
    # Score final avec pondération (vectorisé)
    priority_scores = (
        dr * 0.2 +  # Domain Rating (20%)
        np.minimum(traffic / 10000, 100) * 0.2 +  # Traffic normalisé (20%)
        gap_normalized * 0.3 +  # Gap concurrentiel (30%)
        thematic_scores * 0.3  # Pertinence thématique (30%)
    )
    
    return priority_scores.round(2), competitor_links

# Interface utilisateur
st.sidebar.header("📁 Upload des fichiers")

# Upload des fichiers
ahrefs_domains_file = st.sidebar.file_uploader(
    "Export Ahrefs - Referring Domains",
    type=['csv'],
    help="Export CSV des domaines référents depuis Ahrefs"
)

ahrefs_pages_file = st.sidebar.file_uploader(
    "Export Ahrefs - Referring Pages",
    type=['csv'],
    help="Export CSV des pages référentes depuis Ahrefs"
)

gsc_keywords_file = st.sidebar.file_uploader(
    "Export GSC - Requêtes",
    type=['csv'],
    help="Export CSV des requêtes depuis Google Search Console"
)

gsc_pages_file = st.sidebar.file_uploader(
    "Export GSC - Pages",
    type=['csv'],
    help="Export CSV des pages depuis Google Search Console"
)

strategic_keywords_file = st.sidebar.file_uploader(
    "Mots-clés stratégiques",
    type=['xlsx', 'csv', 'xls'],
    help="Fichier Excel ou CSV contenant vos mots-clés stratégiques"
)

serp_data_file = st.sidebar.file_uploader(
    "Export SERPs (Optionnel)",
    type=['xlsx', 'csv', 'xls'],
    help="Données des SERPs pour les mots-clés stratégiques"
)

# Analyse SERPs - Nouveau
st.sidebar.header("📊 Analyse SERPs (Optionnel)")
serp_files = st.sidebar.file_uploader(
    "Exports SERPs Ahrefs",
    type=['csv'],
    accept_multiple_files=True,
    help="Uploadez jusqu'à 10 fichiers d'exports SERPs d'Ahrefs pour l'analyse micro des mots-clés"
)

# Paramètres de filtrage
st.sidebar.header("🎛️ Paramètres de filtrage")

# Charger et traiter les données
if ahrefs_domains_file is not None:
    
    # Lecture du fichier Ahrefs Domains
    with st.spinner("Chargement des données Ahrefs Domains..."):
        ahrefs_domains_df = read_ahrefs_csv(ahrefs_domains_file)
    
    if ahrefs_domains_df is not None:
        st.success(f"✅ Fichier Ahrefs Domains chargé : {len(ahrefs_domains_df)} domaines")
        
        # Identifier les colonnes des concurrents
        competitor_columns = [col for col in ahrefs_domains_df.columns if col.startswith('www.')]
        main_site = 'www.explore-grandest.com/' if 'www.explore-grandest.com/' in competitor_columns else competitor_columns[0]
        other_competitors = [col for col in competitor_columns if col != main_site]
        
        st.sidebar.write(f"**Site principal détecté :** {main_site}")
        st.sidebar.write(f"**Concurrents détectés :** {len(other_competitors)}")
        
        # Filtre par nombre de concurrents
        max_competitors = len(other_competitors)
        min_competitors_filter = st.sidebar.slider(
            "Minimum de concurrents ayant des liens",
            min_value=1,
            max_value=max_competitors,
            value=1,
            help="Filtrer les domaines qui font des liens vers au moins X concurrents"
        )
        
        # Filtres sur les métriques
        st.sidebar.subheader("Filtres sur les métriques")
        
        min_dr = st.sidebar.slider(
            "Domain Rating minimum",
            min_value=0,
            max_value=100,
            value=20,
            help="Filtrer par Domain Rating minimum"
        )
        
        min_traffic = st.sidebar.number_input(
            "Trafic minimum",
            min_value=0,
            value=1000,
            help="Filtrer par trafic minimum du domaine"
        )
        
        # Charger les autres fichiers pour l'analyse thématique
        keywords_data = None
        pages_data = None
        serp_data = None
        serp_analysis = None
        serp_benchmarks = None
        serp_recommendations = None
        
        if strategic_keywords_file is not None:
            if strategic_keywords_file.name.endswith(('.xlsx', '.xls')):
                keywords_data = pd.read_excel(strategic_keywords_file)
            else:
                keywords_data = pd.read_csv(strategic_keywords_file)
            st.success(f"✅ Mots-clés stratégiques chargés : {len(keywords_data)} mots-clés")
        
        if gsc_pages_file is not None:
            pages_data = pd.read_csv(gsc_pages_file)
            # Nettoyer les données GSC
            if 'CTR' in pages_data.columns:
                pages_data['CTR'] = pages_data['CTR'].apply(clean_percentage)
            st.success(f"✅ Pages GSC chargées : {len(pages_data)} pages")
        
        if gsc_keywords_file is not None:
            gsc_keywords_data = pd.read_csv(gsc_keywords_file)
            if 'CTR' in gsc_keywords_data.columns:
                gsc_keywords_data['CTR'] = gsc_keywords_data['CTR'].apply(clean_percentage)
            st.success(f"✅ Requêtes GSC chargées : {len(gsc_keywords_data)} requêtes")
        
        if serp_data_file is not None:
            if serp_data_file.name.endswith(('.xlsx', '.xls')):
                serp_data = pd.read_excel(serp_data_file)
            else:
                serp_data = pd.read_csv(serp_data_file)
            st.success(f"✅ Données SERPs chargées : {len(serp_data)} entrées")
        
        # Traitement des fichiers SERPs multiples
        if serp_files and len(serp_files) > 0:
            with st.spinner("Analyse des fichiers SERPs..."):
                serp_analysis = analyze_serp_data(serp_files)
                if serp_analysis is not None:
                    serp_benchmarks = calculate_serp_benchmarks(serp_analysis)
                    serp_recommendations = generate_serp_recommendations(serp_benchmarks)
                    st.success(f"✅ Analyse SERPs : {len(serp_files)} fichiers, {len(serp_analysis)} résultats analysés")
                else:
                    st.warning("❌ Aucune donnée SERP valide trouvée")
        
        # Appliquer les filtres et calculer les scores (OPTIMISÉ)
        with st.spinner("Application des filtres et calcul des scores..."):
            filtered_df = ahrefs_domains_df.copy()
            
            # Convertir les colonnes numériques (vectorisé)
            filtered_df['Domain rating'] = pd.to_numeric(filtered_df['Domain rating'], errors='coerce').fillna(0)
            filtered_df['Domain traffic'] = pd.to_numeric(filtered_df['Domain traffic'], errors='coerce').fillna(0)
            
            # Appliquer les filtres (vectorisé)
            mask = (
                (filtered_df['Domain rating'] >= min_dr) &
                (filtered_df['Domain traffic'] >= min_traffic)
            )
            filtered_df = filtered_df[mask]
            
            # Calculer le nombre de liens concurrents (vectorisé)
            competitor_data = filtered_df[other_competitors].fillna(0)
            filtered_df['competitor_links_count'] = (competitor_data > 0).sum(axis=1)
            
            # Filtre nombre de concurrents
            filtered_df = filtered_df[filtered_df['competitor_links_count'] >= min_competitors_filter]
            
            # Calculer les scores de priorité (VERSION OPTIMISÉE)
            if len(filtered_df) > 0:
                priority_scores, competitor_links = calculate_priority_score_vectorized(
                    filtered_df, keywords_data, pages_data
                )
                filtered_df['priority_score'] = priority_scores
                
                # Trier par score de priorité
                filtered_df = filtered_df.sort_values('priority_score', ascending=False)
                
                # Ajouter des métriques calculées (vectorisé)
                filtered_df['gap_opportunity'] = filtered_df['competitor_links_count']
                filtered_df['traffic_potential'] = filtered_df['Domain traffic'] * filtered_df['Domain rating'] / 100
        
        # Traitement des Referring Pages si disponible
        filtered_pages_df = None
        if ahrefs_pages_file is not None:
            with st.spinner("Traitement des referring pages..."):
                ahrefs_pages_df = read_ahrefs_csv(ahrefs_pages_file)
                if ahrefs_pages_df is not None:
                    # Nettoyer et filtrer les pages selon les mêmes critères
                    ahrefs_pages_df['Domain rating'] = pd.to_numeric(ahrefs_pages_df.get('Domain rating', 0), errors='coerce').fillna(0)
                    ahrefs_pages_df['Domain traffic'] = pd.to_numeric(ahrefs_pages_df.get('Domain traffic', 0), errors='coerce').fillna(0)
                    ahrefs_pages_df['Page traffic'] = pd.to_numeric(ahrefs_pages_df.get('Page traffic', 0), errors='coerce').fillna(0)
                    ahrefs_pages_df['UR'] = pd.to_numeric(ahrefs_pages_df.get('UR', 0), errors='coerce').fillna(0)
                    
                    # Filtrer les pages selon les domaines prioritaires
                    priority_domains = set(filtered_df['Domain'].tolist())
                    if 'Referring page URL' in ahrefs_pages_df.columns:
                        ahrefs_pages_df['extracted_domain'] = ahrefs_pages_df['Referring page URL'].apply(
                            lambda x: urlparse(str(x)).netloc.replace('www.', '') if pd.notna(x) else ''
                        )
                        filtered_pages_df = ahrefs_pages_df[ahrefs_pages_df['extracted_domain'].isin(priority_domains)].copy()
                    elif 'Domain' in ahrefs_pages_df.columns:
                        filtered_pages_df = ahrefs_pages_df[ahrefs_pages_df['Domain'].isin(priority_domains)].copy()
                    
                    if filtered_pages_df is not None and len(filtered_pages_df) > 0:
                        # Calculer un score pour les pages
                        filtered_pages_df['page_score'] = (
                            filtered_pages_df['Domain rating'] * 0.3 +
                            np.minimum(filtered_pages_df['Page traffic'] / 1000, 100) * 0.3 +
                            filtered_pages_df['UR'] * 0.4
                        )
                        filtered_pages_df = filtered_pages_df.sort_values('page_score', ascending=False)
        
        # Affichage des résultats avec onglets
        st.header("📊 Résultats de l'analyse")
        
        # Créer les onglets
        tabs_list = ["📈 Tableau de bord", "🎯 Referring Domains"]
        
        if filtered_pages_df is not None and len(filtered_pages_df) > 0:
            tabs_list.append("📄 Referring Pages")
        
        if serp_analysis is not None and len(serp_analysis) > 0:
            tabs_list.append("🎯 Analyse SERPs")
        
        tabs_list.extend(["📁 Fichiers d'entrée", "💾 Export CSV"])
        
        tabs = st.tabs(tabs_list)
        
        with tabs[0]:  # Tableau de bord
            # Métriques principales
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
                    delta=f"Min: {min_dr}"
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
            
            # Graphiques
            st.subheader("📈 Visualisations")
            
            # Graphique scatter plot
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
            
            # Graphique en barres des top domaines
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
            
            # Analyse par segments
            st.subheader("🎯 Analyse par segments")
            
            # Créer des segments basés sur le score
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
            
            # Graphique en secteurs
            priority_counts = filtered_df['priority_category'].value_counts()
            fig_pie = px.pie(
                values=priority_counts.values,
                names=priority_counts.index,
                title="Répartition des opportunités par niveau de priorité"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Recommandations automatiques
            st.subheader("💡 Recommandations automatiques")
            
            # Top 3 domaines prioritaires
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
                    
                    # Afficher quels concurrents ont des liens
                    linked_competitors = []
                    for comp in other_competitors:
                        if pd.notna(domain[comp]) and int(domain[comp]) > 0:
                            linked_competitors.append(f"{comp} ({int(domain[comp])} liens)")
                    
                    if linked_competitors:
                        st.write("**Concurrents ayant des liens :**")
                        st.write(" • ".join(linked_competitors))
        
        with tabs[1]:  # Referring Domains
            # Tableau détaillé des domaines
            st.subheader("📋 Tableau détaillé des opportunités - Referring Domains")
            
            # Sélectionner les colonnes à afficher
            display_columns = [
                'Domain', 'Domain rating', 'Domain traffic', 'priority_score',
                'competitor_links_count', 'gap_opportunity'
            ] + other_competitors[:3]  # Afficher les 3 premiers concurrents
            
            display_df = filtered_df[display_columns].head(100)
            
            # Formatter le tableau
            display_df = display_df.round(2)
            display_df.columns = [
                'Domaine', 'DR', 'Trafic', 'Score Priorité',
                'Nb Concurrents', 'Opportunité Gap'
            ] + [f'Concurrent {i+1}' for i in range(len(other_competitors[:3]))]
            
            # Styling du tableau
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
        
        tab_index = 2
        
        if filtered_pages_df is not None and len(filtered_pages_df) > 0:
            with tabs[tab_index]:  # Referring Pages
                st.subheader("📄 Pages référentes prioritaires à cibler")
                
                # Colonnes à afficher pour les pages
                page_display_columns = []
                available_columns = filtered_pages_df.columns.tolist()
                
                # Colonnes essentielles
                essential_cols = ['Referring page title', 'Referring page URL', 'Domain', 'Domain ratingessential_cols = ['Referring page title', 'Referring page URL', 'Domain', 'Domain rating', 'UR', 'Page traffic', 'page_score']
               for col in essential_cols:
                   if col in available_columns:
                       page_display_columns.append(col)
               
               # Ajouter colonnes concurrents si disponibles
               for comp in other_competitors[:2]:
                   if comp in available_columns:
                       page_display_columns.append(comp)
               
               pages_display_df = filtered_pages_df[page_display_columns].head(200)
               
               # Renommer les colonnes pour plus de clarté
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
               
               # Statistiques sur les pages
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
           
           tab_index += 1
       
       # Onglet Analyse SERPs
       if serp_analysis is not None and len(serp_analysis) > 0:
           with tabs[tab_index]:  # Analyse SERPs
               st.subheader("🎯 Analyse des SERPs - Benchmarks par mot-clé")
               
               if serp_benchmarks and serp_recommendations:
                   # Sélecteur de mot-clé
                   keywords_list = list(serp_benchmarks.keys())
                   selected_keyword = st.selectbox(
                       "Choisissez un mot-clé à analyser",
                       keywords_list,
                       help="Sélectionnez un mot-clé pour voir ses benchmarks détaillés"
                   )
                   
                   if selected_keyword:
                       keyword_data = serp_benchmarks[selected_keyword]
                       keyword_recs = serp_recommendations[selected_keyword]
                       
                       # Métriques du mot-clé
                       st.write(f"**📊 Analyse pour le mot-clé : '{selected_keyword}'**")
                       st.write(f"Nombre de résultats analysés : {keyword_data['total_results']}")
                       
                       # Tableau des benchmarks
                       st.subheader("📈 Benchmarks par position")
                       
                       benchmark_data = []
                       for pos_group, data in keyword_data['positions'].items():
                           position_name = {
                               'top_1': 'Position #1',
                               'top_3': 'Top 3',
                               'top_5': 'Top 5',
                               'top_10': 'Top 10'
                           }.get(pos_group, pos_group)
                           
                           benchmark_data.append({
                               'Position': position_name,
                               'Backlinks (Médian)': int(data['backlinks_median']),
                               'Backlinks (Moyen)': int(data['backlinks_mean']),
                               'Domaines Référents (Médian)': int(data['referring_domains_median']),
                               'DR (Médian)': int(data['domain_rating_median']),
                               'UR (Médian)': int(data['url_rating_median']),
                               'Trafic (Médian)': int(data['traffic_median']),
                               'Échantillon': data['count']
                           })
                       
                       benchmark_df = pd.DataFrame(benchmark_data)
                       st.dataframe(benchmark_df, use_container_width=True)
                       
                       # Recommandations
                       st.subheader("💡 Recommandations pour ce mot-clé")
                       
                       for pos_group, rec_data in keyword_recs['recommendations'].items():
                           with st.expander(f"🎯 Objectif : {rec_data['target']}"):
                               col1, col2, col3 = st.columns(3)
                               
                               with col1:
                                   st.metric("Backlinks cible", rec_data['backlinks_target'])
                                   st.metric("Domaines référents cible", rec_data['referring_domains_target'])
                               
                               with col2:
                                   st.metric("Domain Rating cible", rec_data['domain_rating_target'])
                                   st.metric("URL Rating cible", rec_data['url_rating_target'])
                               
                               with col3:
                                   st.info(rec_data['description'])
                       
                       # Graphiques
                       st.subheader("📊 Visualisations")
                       
                       # Données pour le graphique
                       serp_keyword_data = serp_analysis[serp_analysis['Keyword'] == selected_keyword]
                       
                       if len(serp_keyword_data) > 0:
                           # Graphique scatter : Position vs Backlinks
                           fig_scatter_serp = px.scatter(
                               serp_keyword_data,
                               x='Position',
                               y='Backlinks',
                               size='Domain rating',
                               color='Referring Domains',
                               hover_data=['URL', 'Traffic'],
                               title=f"Position vs Backlinks pour '{selected_keyword}'",
                               labels={
                                   'Position': 'Position dans le SERP',
                                   'Backlinks': 'Nombre de Backlinks',
                                   'Referring Domains': 'Domaines Référents'
                               }
                           )
                           fig_scatter_serp.update_xaxis(dtick=1, range=[0.5, 10.5])
                           st.plotly_chart(fig_scatter_serp, use_container_width=True)
                           
                           # Graphique en barres : DR par position
                           fig_bar_serp = px.box(
                               serp_keyword_data,
                               x='Position',
                               y='Domain rating',
                               title=f"Distribution du Domain Rating par position pour '{selected_keyword}'",
                               labels={
                                   'Position': 'Position dans le SERP',
                                   'Domain rating': 'Domain Rating'
                               }
                           )
                           st.plotly_chart(fig_bar_serp, use_container_width=True)
                   
                   # Vue d'ensemble de tous les mots-clés
                   st.subheader("📋 Vue d'ensemble - Tous les mots-clés")
                   
                   overview_data = []
                   for keyword, data in serp_benchmarks.items():
                       if 'top_3' in data['positions']:
                           top3_data = data['positions']['top_3']
                           overview_data.append({
                               'Mot-clé': keyword,
                               'Résultats analysés': data['total_results'],
                               'Backlinks Top 3 (Médian)': int(top3_data['backlinks_median']),
                               'DR Top 3 (Médian)': int(top3_data['domain_rating_median']),
                               'Domaines Référents Top 3 (Médian)': int(top3_data['referring_domains_median']),
                               'Trafic Top 3 (Médian)': int(top3_data['traffic_median'])
                           })
                   
                   if overview_data:
                       overview_df = pd.DataFrame(overview_data)
                       st.dataframe(overview_df, use_container_width=True)
               
               else:
                   st.warning("Aucune analyse SERP disponible. Vérifiez le format de vos fichiers.")
           
           tab_index += 1
       
       # Onglet Fichiers d'entrée
       with tabs[tab_index]:
           st.subheader("📁 Fichiers d'entrée - Aperçu des données")
           
           # Sous-onglets pour les différents fichiers
           if keywords_data is not None or pages_data is not None or 'gsc_keywords_data' in locals():
               sub_tabs = []
               sub_tab_names = []
               
               if keywords_data is not None:
                   sub_tab_names.append("🎯 Mots-clés stratégiques")
               if 'gsc_keywords_data' in locals() and gsc_keywords_data is not None:
                   sub_tab_names.append("📊 GSC Requêtes")
               if pages_data is not None:
                   sub_tab_names.append("📄 GSC Pages")
               if serp_data is not None:
                   sub_tab_names.append("🔍 SERPs")
               if serp_analysis is not None:
                   sub_tab_names.append("🎯 Analyse SERPs")
               
               if sub_tab_names:
                   sub_tabs = st.tabs(sub_tab_names)
                   
                   tab_idx = 0
                   
                   if keywords_data is not None:
                       with sub_tabs[tab_idx]:
                           st.write(f"**{len(keywords_data)} mots-clés stratégiques chargés**")
                           st.dataframe(keywords_data.head(20), use_container_width=True)
                       tab_idx += 1
                   
                   if 'gsc_keywords_data' in locals() and gsc_keywords_data is not None:
                       with sub_tabs[tab_idx]:
                           st.write(f"**{len(gsc_keywords_data)} requêtes GSC chargées**")
                           st.dataframe(gsc_keywords_data.head(20), use_container_width=True)
                       tab_idx += 1
                   
                   if pages_data is not None:
                       with sub_tabs[tab_idx]:
                           st.write(f"**{len(pages_data)} pages GSC chargées**")
                           st.dataframe(pages_data.head(20), use_container_width=True)
                       tab_idx += 1
                   
                   if serp_data is not None:
                       with sub_tabs[tab_idx]:
                           st.write(f"**{len(serp_data)} entrées SERPs chargées**")
                           st.dataframe(serp_data.head(20), use_container_width=True)
                       tab_idx += 1
                   
                   if serp_analysis is not None:
                       with sub_tabs[tab_idx]:
                           st.write(f"**{len(serp_analysis)} entrées d'analyse SERPs**")
                           st.write(f"**Mots-clés analysés :** {', '.join(serp_analysis['Keyword'].unique()[:5])}...")
                           st.dataframe(serp_analysis.head(20), use_container_width=True)
           else:
               st.info("Aucun fichier optionnel chargé. Uploadez vos fichiers GSC et mots-clés stratégiques pour enrichir l'analyse.")
       
       # Onglet Export CSV
       tab_index += 1
       with tabs[tab_index]:
           # Export des résultats
           st.subheader("💾 Télécharger les résultats")
           
           if len(filtered_df) > 0:
               # Préparer le DataFrame final pour export
               export_df = filtered_df.copy()
               
               # Renommer les colonnes pour plus de clarté
               export_columns = {
                   'Domain': 'Domaine',
                   'Domain rating': 'Domain_Rating',
                   'Domain traffic': 'Trafic_Mensuel',
                   'priority_score': 'Score_Priorite',
                   'competitor_links_count': 'Nb_Concurrents_Lies',
                   'gap_opportunity': 'Opportunite_Gap',
                   'traffic_potential': 'Potentiel_Trafic'
               }
               
               # Ajouter les colonnes des concurrents avec des noms plus clairs
               for i, comp in enumerate(other_competitors):
                   export_columns[comp] = f'Concurrent_{i+1}_Liens'
               
               export_df = export_df.rename(columns=export_columns)
               
               # Sélectionner et ordonner les colonnes importantes
               key_columns = [
                   'Domaine', 'Domain_Rating', 'Trafic_Mensuel', 'Score_Priorite',
                   'Nb_Concurrents_Lies', 'Opportunite_Gap', 'Potentiel_Trafic'
               ]
               
               # Ajouter les colonnes concurrents
               competitor_columns_renamed = [f'Concurrent_{i+1}_Liens' for i in range(len(other_competitors))]
               final_columns = key_columns + competitor_columns_renamed
               
               # Créer le DataFrame final
               final_export_df = export_df[final_columns].round(2)
               
               # Informations sur l'analyse
               st.write(f"**Nombre de domaines analysés :** {len(filtered_df)}")
               st.write(f"**Score de priorité moyen :** {filtered_df['priority_score'].mean():.2f}/100")
               st.write(f"**Domain Rating moyen :** {filtered_df['Domain rating'].mean():.1f}")
               
               # Boutons de téléchargement
               col1, col2 = st.columns(2)
               
               with col1:
                   st.write("**🎯 Referring Domains**")
                   
                   # Bouton de téléchargement principal
                   csv_data = final_export_df.to_csv(index=False, encoding='utf-8')
                   
                   st.download_button(
                       label="📄 Analyse complète (CSV)",
                       data=csv_data,
                       file_name=f"audit_netlinking_domains_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv",
                       use_container_width=True
                   )
                   
                   # Top 50
                   top_50 = final_export_df.head(50)
                   csv_top_50 = top_50.to_csv(index=False, encoding='utf-8')
                   st.download_button(
                       label="🥇 Top 50 prioritaires",
                       data=csv_top_50,
                       file_name=f"top_50_domains_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                       mime="text/csv",
                       use_container_width=True
                   )
               
               with col2:
                   if filtered_pages_df is not None and len(filtered_pages_df) > 0:
                       st.write("**📄 Referring Pages**")
                       
                       # Préparer l'export des pages
                       pages_export_df = filtered_pages_df.copy()
                       pages_export_columns = {}
                       
                       # Renommer les colonnes importantes
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
                       
                       # Export complet des pages
                       pages_csv = pages_export_df.to_csv(index=False, encoding='utf-8')
                       st.download_button(
                           label="📄 Pages complètes (CSV)",
                           data=pages_csv,
                           file_name=f"referring_pages_complete_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv",
                           use_container_width=True
                       )
                       
                       # Top 100 pages
                       top_100_pages = pages_export_df.head(100)
                       pages_top_csv = top_100_pages.to_csv(index=False, encoding='utf-8')
                       st.download_button(
                           label="🥇 Top 100 pages",
                           data=pages_top_csv,
                           file_name=f"top_100_pages_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv",
                           use_container_width=True
                       )
                   else:
                       st.write("**📄 Referring Pages**")
                       st.info("Uploadez le fichier Ahrefs Referring Pages pour obtenir l'analyse des pages.")
               
               # Exports spécialisés
               st.write("**🎯 Exports spécialisés**")
               col3, col4, col5 = st.columns(3)
               
               with col3:
                   high_priority = final_export_df[final_export_df['Score_Priorite'] >= 70]
                   if len(high_priority) > 0:
                       csv_high_priority = high_priority.to_csv(index=False, encoding='utf-8')
                       st.download_button(
                           label=f"🔥 Priorité max ({len(high_priority)})",
                           data=csv_high_priority,
                           file_name=f"priorite_max_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv",
                           use_container_width=True
                       )
                   else:
                       st.button("Aucun domaine priorité max", disabled=True, use_container_width=True)
               
               with col4:
                   high_gap = final_export_df[final_export_df['Nb_Concurrents_Lies'] >= max_competitors - 1]
                   if len(high_gap) > 0:
                       csv_high_gap = high_gap.to_csv(index=False, encoding='utf-8')
                       st.download_button(
                           label=f"⚡ Gaps importants ({len(high_gap)})",
                           data=csv_high_gap,
                           file_name=f"gaps_importants_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv",
                           use_container_width=True
                       )
                   else:
                       st.button("Aucun gap important", disabled=True, use_container_width=True)
               
               with col5:
                   high_dr = final_export_df[final_export_df['Domain_Rating'] >= 70]
                   if len(high_dr) > 0:
                       csv_high_dr = high_dr.to_csv(index=False, encoding='utf-8')
                       st.download_button(
                           label=f"⭐ DR élevé ({len(high_dr)})",
                           data=csv_high_dr,
                           file_name=f"dr_eleve_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv",
                           use_container_width=True
                       )
                   else:
                       st.button("Aucun DR élevé", disabled=True, use_container_width=True)
               
               # Export des analyses SERPs
               if serp_benchmarks and serp_recommendations:
                   st.write("**🎯 Analyse SERPs**")
                   
                   # Préparer les données d'export SERPs
                   serp_export_data = []
                   
                   for keyword, data in serp_benchmarks.items():
                       keyword_recs = serp_recommendations.get(keyword, {}).get('recommendations', {})
                       
                       # Données générales du mot-clé
                       base_data = {
                           'Mot_Cle': keyword,
                           'Nb_Resultats_Analyses': data['total_results']
                       }
                       
                       # Ajouter les benchmarks pour chaque position
                       positions_data = {}
                       for pos_group, pos_data in data['positions'].items():
                           prefix = {
                               'top_1': 'Position_1',
                               'top_3': 'Top_3',
                               'top_5': 'Top_5',
                               'top_10': 'Top_10'
                           }.get(pos_group, pos_group)
                           
                           positions_data.update({
                               f'{prefix}_Backlinks_Median': int(pos_data['backlinks_median']),
                               f'{prefix}_Backlinks_Moyen': int(pos_data['backlinks_mean']),
                               f'{prefix}_Domaines_Referents_Median': int(pos_data['referring_domains_median']),
                               f'{prefix}_DR_Median': int(pos_data['domain_rating_median']),
                               f'{prefix}_UR_Median': int(pos_data['url_rating_median']),
                               f'{prefix}_Trafic_Median': int(pos_data['traffic_median'])
                           })
                       
                       # Ajouter les recommandations
                       recommendations_data = {}
                       for pos_group, rec_data in keyword_recs.items():
                           prefix = {
                               'top_1': 'Recommandation_Position_1',
                               'top_3': 'Recommandation_Top_3',
                               'top_5': 'Recommandation_Top_5'
                           }.get(pos_group, f'Recommandation_{pos_group}')
                           
                           recommendations_data.update({
                               f'{prefix}_Backlinks_Cible': rec_data['backlinks_target'],
                               f'{prefix}_DR_Cible': rec_data['domain_rating_target'],
                               f'{prefix}_Domaines_Referents_Cible': rec_data['referring_domains_target'],
                               f'{prefix}_Description': rec_data['description']
                           })
                       
                       # Combiner toutes les données
                       complete_data = {**base_data, **positions_data, **recommendations_data}
                       serp_export_data.append(complete_data)
                   
                   if serp_export_data:
                       serp_export_df = pd.DataFrame(serp_export_data)
                       serp_csv = serp_export_df.to_csv(index=False, encoding='utf-8')
                       
                       st.download_button(
                           label="📊 Analyse SERPs complète (CSV)",
                           data=serp_csv,
                           file_name=f"analyse_serps_benchmarks_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv",
                           use_container_width=True
                       )
                       
                       # Export simplifié des recommandations
                       simplified_recs = []
                       for keyword, recs in serp_recommendations.items():
                           if 'top_3' in recs['recommendations']:
                               top3_rec = recs['recommendations']['top_3']
                               simplified_recs.append({
                                   'Mot_Cle': keyword,
                                   'Objectif': 'Top 3',
                                   'Backlinks_Necessaires': top3_rec['backlinks_target'],
                                   'DR_Necessaire': top3_rec['domain_rating_target'],
                                   'Domaines_Referents_Necessaires': top3_rec['referring_domains_target'],
                                   'Recommandation': top3_rec['description']
                               })
                       
                       if simplified_recs:
                           simplified_df = pd.DataFrame(simplified_recs)
                           simplified_csv = simplified_df.to_csv(index=False, encoding='utf-8')
                           
                           st.download_button(
                               label="🎯 Recommandations Top 3 (CSV)",
                               data=simplified_csv,
                               file_name=f"recommandations_top3_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv",
                               use_container_width=True
                           )
           else:
               st.warning("Aucun domaine ne correspond aux critères de filtrage sélectionnés.")

else:
   # Page d'accueil sans fichiers
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
       
       4. **[NOUVEAU] Exportez vos SERPs depuis Ahrefs (Optionnel) :**
          - Allez dans "Keywords Explorer" > "SERP overview"
          - Exportez le top 10 pour vos mots-clés prioritaires
          - Uploadez jusqu'à 10 fichiers pour l'analyse micro
       
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
       - **[NOUVEAU] Analyse micro des SERPs** : Benchmarks par mot-clé pour définir vos objectifs de netlinking
       - **Fournit des analyses complètes** avec tableaux de bord, graphiques et exports CSV
       """)
   
   with st.expander("📊 Résultats obtenus"):
       st.markdown("""
       - Tableau de bord avec graphiques interactifs
       - Liste des domaines prioritaires à contacter
       - Liste des pages référentes spécifiques à cibler
       - **[NOUVEAU] Analyse SERPs** : Benchmarks et recommandations par mot-clé
       - Fichiers CSV structurés pour vos campagnes
       - Aperçu de tous vos fichiers d'entrée
       """)
   
   # Afficher un exemple de structure attendue
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
       
       **[NOUVEAU] Ahrefs - SERPs Overview :**
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
       """)

# Footer
st.markdown("---")
st.markdown("**Développé par [JC Espinosa](https://jc-espinosa.com) pour optimiser vos campagnes de netlinking SEO
