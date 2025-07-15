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

def analyze_serp_data(serp_files):
    """Analyse les fichiers SERP pour générer des benchmarks"""
    serp_analysis_results = {}
    
    for serp_file in serp_files:
        try:
            # Lire le fichier SERP (format Ahrefs UTF-16LE)
            content = serp_file.read()
            decoded_content = content.decode('utf-16le')
            decoded_content = decoded_content.replace('\x00', '').replace('\ufeff', '')
            
            serp_df = pd.read_csv(StringIO(decoded_content), sep='\t')
            
            if len(serp_df) == 0:
                continue
                
            # Identifier le mot-clé principal
            keyword = serp_df['Keyword'].iloc[0] if 'Keyword' in serp_df.columns else f"Mot-clé_{len(serp_analysis_results)+1}"
            
            # Convertir les colonnes numériques
            numeric_columns = ['Position', 'Backlinks', 'Referring Domains', 'Domain rating', 'URL rating', 'Traffic']
            for col in numeric_columns:
                if col in serp_df.columns:
                    serp_df[col] = pd.to_numeric(serp_df[col], errors='coerce')
            
            # Filtrer le top 10
            top_10 = serp_df[serp_df['Position'] <= 10].copy()
            
            if len(top_10) == 0:
                continue
            
            # Calculer les benchmarks par segments
            analysis = {
                'keyword': keyword,
                'total_results': len(top_10),
                'segments': {}
            }
            
            # Définir les segments
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
            
            # Ajouter les données détaillées
            detail_columns = ['Position', 'URL', 'Backlinks', 'Referring Domains', 'Domain rating', 'URL rating', 'Traffic', 'Title']
            available_columns = [col for col in detail_columns if col in top_10.columns]
            analysis['detailed_data'] = top_10[available_columns].to_dict('records')
            
            serp_analysis_results[keyword] = analysis
            
        except Exception as e:
            st.warning(f"Erreur lors de l'analyse du fichier {serp_file.name}: {str(e)}")
            continue
    
    return serp_analysis_results

def generate_serp_recommendations(serp_analysis):
    """Génère des recommandations basées sur l'analyse SERP"""
    recommendations = {}
    
    for keyword, analysis in serp_analysis.items():
        reco = {
            'keyword': keyword,
            'recommendations': []
        }
        
        # Recommandations pour le top 1
        if 'top_1' in analysis['segments']:
            top1_data = analysis['segments']['top_1']
            reco['recommendations'].append({
                'position_target': 'Position #1',
                'backlinks_target': int(top1_data['backlinks_median']),
                'rd_target': int(top1_data['rd_median']),
                'dr_target': int(top1_data['dr_median']),
                'ur_target': int(top1_data['ur_median']),
                'message': f"Pour viser la 1ère position sur '{keyword}': {int(top1_data['backlinks_median'])} backlinks, {int(top1_data['rd_median'])} RD, DR {int(top1_data['dr_median'])}"
            })
        
        # Recommandations pour le top 3
        if 'top_3' in analysis['segments']:
            top3_data = analysis['segments']['top_3']
            reco['recommendations'].append({
                'position_target': 'Top 3',
                'backlinks_target': int(top3_data['backlinks_median']),
                'rd_target': int(top3_data['rd_median']),
                'dr_target': int(top3_data['dr_median']),
                'ur_target': int(top3_data['ur_median']),
                'message': f"Pour être dans le top 3 sur '{keyword}': {int(top3_data['backlinks_median'])} backlinks, {int(top3_data['rd_median'])} RD, DR {int(top3_data['dr_median'])}"
            })
        
        # Recommandations pour le top 5
        if 'top_5' in analysis['segments']:
            top5_data = analysis['segments']['top_5']
            reco['recommendations'].append({
                'position_target': 'Top 5',
                'backlinks_target': int(top5_data['backlinks_median']),
                'rd_target': int(top5_data['rd_median']),
                'dr_target': int(top5_data['dr_median']),
                'ur_target': int(top5_data['ur_median']),
                'message': f"Pour être dans le top 5 sur '{keyword}': {int(top5_data['backlinks_median'])} backlinks, {int(top5_data['rd_median'])} RD, DR {int(top5_data['dr_median'])}"
            })
        
        recommendations[keyword] = reco
    
    return recommendations

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

serp_analysis_files = st.sidebar.file_uploader(
    "Analyse SERPs (Optionnel)",
    type=['csv'],
    accept_multiple_files=True,
    help="Fichiers CSV d'analyse SERP Ahrefs (10 maximum) - Un fichier par mot-clé"
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
        serp_analysis_data = None
        
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
        
        if serp_analysis_files and len(serp_analysis_files) > 0:
            with st.spinner("Analyse des fichiers SERP..."):
                serp_analysis_data = analyze_serp_data(serp_analysis_files)
                if serp_analysis_data:
                    st.success(f"✅ Analyse SERP : {len(serp_analysis_data)} mots-clés analysés")
                else:
                    st.warning("Aucune donnée SERP valide trouvée")
        
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
        
        # Créer les onglets selon les données disponibles
        tab_names = ["📈 Tableau de bord", "🎯 Referring Domains"]
        
        if filtered_pages_df is not None and len(filtered_pages_df) > 0:
            tab_names.append("📄 Referring Pages")
        
        if serp_analysis_data:
            tab_names.append("🎯 Analyse SERPs")
        
        tab_names.extend(["📁 Fichiers d'entrée", "💾 Export CSV"])
        
        tabs = st.tabs(tab_names)
        
        # Gestion dynamique des onglets
        tab_index = 0
        
        # Onglet Tableau de bord
        with tabs[tab_index]:
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
        
        # Onglet Referring Domains
        tab_index += 1
        with tabs[tab_index]:
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
                        format="%d
