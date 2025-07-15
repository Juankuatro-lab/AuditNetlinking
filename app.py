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
                        format="%d",                    ),
                    "Trafic": st.column_config.NumberColumn(
                        "Trafic",
                        help="Trafic mensuel estimé",
                        format="%d",
                    ),
                }
            )
        
        # Onglet Referring Pages (si disponible)
        if filtered_pages_df is not None and len(filtered_pages_df) > 0:
            tab_index += 1
            with tabs[tab_index]:
                st.subheader("📄 Pages référentes prioritaires à cibler")
                
                # Colonnes à afficher pour les pages
                page_display_columns = []
                available_columns = filtered_pages_df.columns.tolist()
                
                # Colonnes essentielles
                essential_cols = ['Referring page title', 'Referring page URL', 'Domain', 'Domain rating', 'UR', 'Page traffic', 'page_score']
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
        
        # Onglet Analyse SERPs (si disponible)
        if serp_analysis_data:
            tab_index += 1
            with tabs[tab_index]:
                st.subheader("🎯 Analyse des SERPs - Benchmarks par mot-clé")
                
                # Sélecteur de mot-clé
                selected_keyword = st.selectbox(
                    "Choisissez un mot-clé à analyser",
                    options=list(serp_analysis_data.keys()),
                    key="serp_keyword_selector"
                )
                
                if selected_keyword:
                    analysis = serp_analysis_data[selected_keyword]
                    
                    # Métriques principales
                    st.write(f"**Analyse pour le mot-clé : {selected_keyword}**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Résultats analysés", analysis['total_results'])
                    
                    if 'top_1' in analysis['segments']:
                        with col2:
                            top1_bl = analysis['segments']['top_1']['backlinks_median']
                            st.metric("Backlinks #1", f"{int(top1_bl)}")
                        
                        with col3:
                            top1_dr = analysis['segments']['top_1']['dr_median']
                            st.metric("DR #1", f"{int(top1_dr)}")
                        
                        with col4:
                            top1_rd = analysis['segments']['top_1']['rd_median']
                            st.metric("RD #1", f"{int(top1_rd)}")
                    
                    # Tableau des benchmarks
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
                            'Backlinks (médian)': int(segment_data['backlinks_median']),
                            'Backlinks (moyen)': int(segment_data['backlinks_mean']),
                            'RD (médian)': int(segment_data['rd_median']),
                            'RD (moyen)': int(segment_data['rd_mean']),
                            'DR (médian)': int(segment_data['dr_median']),
                            'DR (moyen)': int(segment_data['dr_mean']),
                            'UR (médian)': int(segment_data['ur_median']),
                            'Trafic (médian)': int(segment_data['traffic_median'])
                        })
                    
                    benchmark_df = pd.DataFrame(benchmark_data)
                    st.dataframe(benchmark_df, use_container_width=True)
                    
                    # Graphique des benchmarks
                    st.subheader("📈 Visualisation des benchmarks")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Graphique Backlinks par segment
                        fig_bl = px.bar(
                            benchmark_df,
                            x='Segment',
                            y='Backlinks (médian)',
                            title="Backlinks médians par segment",
                            text='Backlinks (médian)'
                        )
                        fig_bl.update_traces(texttemplate='%{text}', textposition='outside')
                        st.plotly_chart(fig_bl, use_container_width=True)
                    
                    with col2:
                        # Graphique DR par segment
                        fig_dr = px.bar(
                            benchmark_df,
                            x='Segment',
                            y='DR (médian)',
                            title="Domain Rating médian par segment",
                            text='DR (médian)',
                            color='DR (médian)',
                            color_continuous_scale='Viridis'
                        )
                        fig_dr.update_traces(texttemplate='%{text}', textposition='outside')
                        st.plotly_chart(fig_dr, use_container_width=True)
                    
                    # Recommandations automatiques
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
                    
                    # Détail du top 10
                    st.subheader("🔍 Détail du top 10")
                    
                    detailed_df = pd.DataFrame(analysis['detailed_data'])
                    if not detailed_df.empty:
                        # Formater le DataFrame pour l'affichage
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
        
        # Onglet Fichiers d'entrée
        tab_index += 1
        with tabs[tab_index]:
            st.subheader("📁 Fichiers d'entrée - Aperçu des données")
            
            # Sous-onglets pour les différents fichiers
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
                        # Afficher un résumé des mots-clés analysés
                        summary_data = []
                        for keyword, analysis in serp_analysis_data.items():
                            summary_data.append({
                                'Mot-clé': keyword,
                                'Résultats': analysis['total_results'],
                                'DR #1': int(analysis['segments']['top_1']['dr_median']) if 'top_1' in analysis['segments'] else 'N/A',
                                'Backlinks #1': int(analysis['segments']['top_1']['backlinks_median']) if 'top_1' in analysis['segments'] else 'N/A'
                            })
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
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
                
                # Export SERP Analysis si disponible
                if serp_analysis_data:
                    st.write("**🎯 Analyse SERPs**")
                    
                    # Créer le DataFrame d'export pour l'analyse SERP
                    serp_export_data = []
                    recommendations_export_data = []
                    
                    for keyword, analysis in serp_analysis_data.items():
                        # Export des benchmarks
                        for segment_name, segment_data in analysis['segments'].items():
                            serp_export_data.append({
                                'Mot_Cle': keyword,
                                'Segment': segment_name,
                                'Nombre_Sites': segment_data['count'],
                                'Backlinks_Median': int(segment_data['backlinks_median']),
                                'Backlinks_Moyen': int(segment_data['backlinks_mean']),
                                'RD_Median': int(segment_data['rd_median']),
                                'RD_Moyen': int(segment_data['rd_mean']),
                                'DR_Median': int(segment_data['dr_median']),
                                'DR_Moyen': int(segment_data['dr_mean']),
                                'UR_Median': int(segment_data['ur_median']),
                                'UR_Moyen': int(segment_data['ur_mean']),
                                'Trafic_Median': int(segment_data['traffic_median']),
                                'Trafic_Moyen': int(segment_data['traffic_mean'])
                            })
                        
                        # Export des recommandations
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
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if serp_export_data:
                            serp_benchmarks_df = pd.DataFrame(serp_export_data)
                            serp_csv = serp_benchmarks_df.to_csv(index=False, encoding='utf-8')
                            st.download_button(
                                label="📊 Benchmarks SERPs",
                                data=serp_csv,
                                file_name=f"serp_benchmarks_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                    
                    with col2:
                        if recommendations_export_data:
                            recommendations_df = pd.DataFrame(recommendations_export_data)
                            reco_csv = recommendations_df.to_csv(index=False, encoding='utf-8')
                            st.download_button(
                                label="💡 Recommandations",
                                data=reco_csv,
                                file_name=f"serp_recommendations_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
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
        """)

# Footer
st.markdown("---")
st.markdown("**Développé par [JC Espinosa](https://jc-espinosa.com) pour optimiser vos campagnes de netlinking SEO**")
