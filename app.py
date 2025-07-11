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
        if filtered_pages_df is not None and len(filtered_pages_df) > 0:
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📈 Tableau de bord", 
                "🎯 Referring Domains", 
                "📄 Referring Pages", 
                "📁 Fichiers d'entrée",
                "💾 Export CSV"
            ])
        else:
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Tableau de bord", 
                "🎯 Referring Domains", 
                "📁 Fichiers d'entrée",
                "💾 Export CSV"
            ])
        
        with tab1:
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
        
        with tab2:
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
        
        if filtered_pages_df is not None and len(filtered_pages_df) > 0:
            with tab3:
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
        
        with tab4:
            st.subheader("📁 Fichiers d'entrée - Aperçu des données")
            
            # Sous-onglets pour les différents fichiers
            if keywords_data is not None or pages_data is not None or gsc_keywords_data is not None:
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
            else:
                st.info("Aucun fichier optionnel chargé. Uploadez vos fichiers GSC et mots-clés stratégiques pour enrichir l'analyse.")
        
        with (tab5 if filtered_pages_df is not None and len(filtered_pages_df) > 0 else tab4):
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
            else:
                st.warning("Aucun domaine ne correspond aux critères de filtrage sélectionnés.")Concurrents_Lies',
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
            
            # Bouton de téléchargement principal
            csv_data = final_export_df.to_csv(index=False, encoding='utf-8')
            
            st.download_button(
                label="Télécharger l'analyse complète (CSV)",
                data=csv_data,
                file_name=f"audit_netlinking_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            # Options de téléchargement par segment
            col1, col2, col3 = st.columns(3)
            
            with col1:
                top_50 = final_export_df.head(50)
                csv_top_50 = top_50.to_csv(index=False, encoding='utf-8')
                st.download_button(
                    label="Top 50 prioritaires",
                    data=csv_top_50,
                    file_name=f"top_50_netlinking_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                high_priority = final_export_df[final_export_df['Score_Priorite'] >= 70]
                if len(high_priority) > 0:
                    csv_high_priority = high_priority.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label=f"Priorité maximale ({len(high_priority)})",
                        data=csv_high_priority,
                        file_name=f"priorite_max_netlinking_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.button(
                        "Aucun domaine priorité max",
                        disabled=True,
                        use_container_width=True
                    )
            
            with col3:
                high_gap = final_export_df[final_export_df['Nb_Concurrents_Lies'] >= max_competitors - 1]
                if len(high_gap) > 0:
                    csv_high_gap = high_gap.to_csv(index=False, encoding='utf-8')
                    st.download_button(
                        label=f"Gaps importants ({len(high_gap)})",
                        data=csv_high_gap,
                        file_name=f"gaps_importants_netlinking_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.button(
                        "Aucun gap important",
                        disabled=True,
                        use_container_width=True
                    )
        else:
            st.warning("Aucun domaine ne correspond aux critères de filtrage sélectionnés.")

else:
    # Page d'accueil sans fichiers
    st.markdown("""
    ## Comment utiliser cet outil ?
    
    ### Étapes à suivre :
    
    1. **Exportez vos données depuis Ahrefs :**
       - Allez dans l'outil "Link Intersect"
       - Ajoutez votre site + vos concurrents
       - Exportez les "Referring Domains" et "Referring Pages"
    
    2. **Exportez vos données depuis Google Search Console :**
       - Allez dans "Performances" > "Requêtes"
       - Exportez les données des requêtes et des pages
    
    3. **Préparez vos mots-clés stratégiques :**
       - Format Excel ou CSV avec colonnes : Keyword, Search Volume, Keyword Difficulty
    
    4. **Uploadez tous les fichiers** dans la barre latérale
    
    5. **Configurez les filtres** selon vos besoins
    
    ### Ce que fait l'outil :
    
    - **Analyse les gaps concurrentiels** : Identifie les sites qui font des liens vers vos concurrents mais pas vers vous
    - **Calcule un score de priorité** basé sur :
      - Domain Rating (20%)
      - Trafic du domaine (20%)  
      - Gap concurrentiel (30%)
      - Pertinence thématique (30%)
    - **Fournit un fichier CSV** avec les résultats priorisés pour vos campagnes de netlinking
    
    ### Résultats obtenus :
    
    - Fichier CSV avec les domaines priorisés
    - Score de priorité pour chaque domaine
    - Métriques détaillées (DR, trafic, gaps concurrentiels)
    - Données segmentées par niveau de priorité
    
    **Commencez par uploader votre export Ahrefs "Referring Domains" dans la barre latérale !**
    """)
    
    # Afficher un exemple de structure attendue
    st.subheader("Structure des fichiers attendus")
    
    with st.expander("Voir les formats de fichiers attendus"):
        st.markdown("""
        **Ahrefs - Referring Domains :**
        ```
        Domain | Domain rating | Domain traffic | Intersect | www.monsite.com | www.concurrent1.com | ...
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
