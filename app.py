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
    page_title="Outil d'Analyse Netlinking SEO",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔗 Outil d'Analyse Netlinking SEO")
st.markdown("**Priorisez vos campagnes de netlinking en croisant les données Ahrefs et GSC**")

# Fonctions utilitaires
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

def extract_domain(url):
    """Extrait le domaine d'une URL"""
    try:
        return urlparse(url).netloc.lower().replace('www.', '')
    except:
        return url

def clean_percentage(value):
    """Nettoie les pourcentages"""
    if isinstance(value, str):
        return float(value.replace('%', '').replace(',', '.'))
    return value

def calculate_thematic_relevance(domain_or_url, keywords_data, pages_data):
    """Calcule la pertinence thématique d'un domaine/URL"""
    relevance_score = 0
    
    # Extraire les mots-clés du domaine/URL
    domain_words = re.findall(r'\w+', domain_or_url.lower())
    
    # Comparer avec les mots-clés stratégiques
    if keywords_data is not None:
        for _, row in keywords_data.iterrows():
            keyword = str(row.get('Keyword', '')).lower()
            keyword_words = re.findall(r'\w+', keyword)
            
            # Calculer la similarité
            common_words = set(domain_words) & set(keyword_words)
            if common_words:
                relevance_score += len(common_words) * row.get('Search Volume', 0) / 1000
    
    # Comparer avec les pages performantes GSC
    if pages_data is not None:
        for _, row in pages_data.iterrows():
            page_url = str(row.get('Pages les plus populaires', '')).lower()
            page_words = re.findall(r'\w+', page_url)
            
            common_words = set(domain_words) & set(page_words)
            if common_words:
                relevance_score += len(common_words) * row.get('Clics', 0) / 100
    
    return min(relevance_score, 100)  # Plafonner à 100

def calculate_priority_score(row, keywords_data=None, pages_data=None):
    """Calcule le score de priorité pour un backlink"""
    
    # Métriques de base
    dr = float(row.get('Domain rating', 0))
    traffic = float(row.get('Domain traffic', 0)) if pd.notna(row.get('Domain traffic', 0)) else 0
    
    # Calcul du gap concurrentiel (nombre de concurrents qui reçoivent des liens)
    competitor_columns = [col for col in row.index if col.startswith('www.') and col != 'www.explore-grandest.com/']
    gap_score = 0
    for col in competitor_columns:
        if pd.notna(row[col]) and int(row[col]) > 0:
            gap_score += 1
    
    # Normaliser le gap (sur 100)
    gap_normalized = (gap_score / len(competitor_columns)) * 100 if competitor_columns else 0
    
    # Calcul de la pertinence thématique
    domain = row.get('Domain', '')
    thematic_score = calculate_thematic_relevance(domain, keywords_data, pages_data)
    
    # Score final avec pondération
    priority_score = (
        dr * 0.2 +  # Domain Rating (20%)
        min(traffic / 10000, 100) * 0.2 +  # Traffic normalisé (20%)
        gap_normalized * 0.3 +  # Gap concurrentiel (30%)
        thematic_score * 0.3  # Pertinence thématique (30%)
    )
    
    return round(priority_score, 2)

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
    type=['xlsx', 'csv'],
    help="Fichier Excel ou CSV contenant vos mots-clés stratégiques"
)

serp_data_file = st.sidebar.file_uploader(
    "Export SERPs (Optionnel)",
    type=['xlsx', 'csv'],
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
            if strategic_keywords_file.name.endswith('.xlsx'):
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
            if serp_data_file.name.endswith('.xlsx'):
                serp_data = pd.read_excel(serp_data_file)
            else:
                serp_data = pd.read_csv(serp_data_file)
            st.success(f"✅ Données SERPs chargées : {len(serp_data)} entrées")
        
        # Appliquer les filtres
        with st.spinner("Application des filtres et calcul des scores..."):
            filtered_df = ahrefs_domains_df.copy()
            
            # Convertir les colonnes numériques
            filtered_df['Domain rating'] = pd.to_numeric(filtered_df['Domain rating'], errors='coerce')
            filtered_df['Domain traffic'] = pd.to_numeric(filtered_df['Domain traffic'], errors='coerce')
            
            # Filtre DR
            filtered_df = filtered_df[filtered_df['Domain rating'] >= min_dr]
            
            # Filtre trafic
            filtered_df = filtered_df[filtered_df['Domain traffic'] >= min_traffic]
            
            # Filtre nombre de concurrents
            def count_competitor_links(row):
                count = 0
                for col in other_competitors:
                    if pd.notna(row[col]) and int(row[col]) > 0:
                        count += 1
                return count
            
            filtered_df['competitor_links_count'] = filtered_df.apply(count_competitor_links, axis=1)
            filtered_df = filtered_df[filtered_df['competitor_links_count'] >= min_competitors_filter]
            
            # Calculer les scores de priorité
            filtered_df['priority_score'] = filtered_df.apply(
                lambda row: calculate_priority_score(row, keywords_data, pages_data), 
                axis=1
            )
            
            # Trier par score de priorité
            filtered_df = filtered_df.sort_values('priority_score', ascending=False)
            
            # Ajouter des métriques calculées
            filtered_df['gap_opportunity'] = filtered_df['competitor_links_count']
            filtered_df['traffic_potential'] = filtered_df['Domain traffic'] * filtered_df['Domain rating'] / 100
        
        # Affichage des résultats
        st.header("📊 Résultats de l'analyse")
        
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
        
        # Tableau détaillé
        st.subheader("📋 Tableau détaillé des opportunités")
        
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
        
        # Export des résultats
        st.subheader("💾 Export des résultats")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 Télécharger le rapport complet (CSV)"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="⬇️ Télécharger CSV",
                    data=csv,
                    file_name=f"netlinking_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            top_n = st.selectbox("Nombre de top domaines à exporter", [10, 20, 50, 100], index=1)
            if st.button(f"📋 Télécharger Top {top_n}"):
                top_csv = filtered_df.head(top_n).to_csv(index=False)
                st.download_button(
                    label=f"⬇️ Télécharger Top {top_n}",
                    data=top_csv,
                    file_name=f"top_{top_n}_netlinking_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        # Analyse détaillée par segments
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
        
        # Statistiques par segment
        stats_by_priority = filtered_df.groupby('priority_category').agg({
            'Domain rating': 'mean',
            'Domain traffic': 'mean',
            'competitor_links_count': 'mean',
            'Domain': 'count'
        }).round(2)
        
        stats_by_priority.columns = ['DR Moyen', 'Trafic Moyen', 'Gap Moyen', 'Nombre de Domaines']
        st.dataframe(stats_by_priority, use_container_width=True)
        
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
        
        # Analyse des gaps les plus importants
        st.write("**🎯 Analyse des gaps concurrentiels :**")
        
        high_gap_domains = filtered_df[filtered_df['competitor_links_count'] >= max_competitors - 1]
        if len(high_gap_domains) > 0:
            st.info(f"📈 {len(high_gap_domains)} domaines font des liens vers {max_competitors - 1}+ concurrents mais pas vers vous !")
            
            gap_sample = high_gap_domains.head(5)[['Domain', 'Domain rating', 'competitor_links_count', 'priority_score']]
            st.dataframe(gap_sample, use_container_width=True)
        
        # Recommandations par DR
        high_dr_domains = filtered_df[filtered_df['Domain rating'] >= 70]
        if len(high_dr_domains) > 0:
            st.success(f"🌟 {len(high_dr_domains)} domaines avec un DR élevé (70+) identifiés")
        
        medium_dr_domains = filtered_df[(filtered_df['Domain rating'] >= 40) & (filtered_df['Domain rating'] < 70)]
        if len(medium_dr_domains) > 0:
            st.warning(f"⚖️ {len(medium_dr_domains)} domaines avec un DR moyen (40-70) - bon rapport effort/bénéfice")

else:
    # Page d'accueil sans fichiers
    st.markdown("""
    ## 🚀 Comment utiliser cet outil ?
    
    ### 📋 Étapes à suivre :
    
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
    
    ### 🎯 Ce que fait l'outil :
    
    - **Analyse les gaps concurrentiels** : Identifie les sites qui font des liens vers vos concurrents mais pas vers vous
    - **Calcule un score de priorité** basé sur :
      - Domain Rating (20%)
      - Trafic du domaine (20%)  
      - Gap concurrentiel (30%)
      - Pertinence thématique (30%)
    - **Fournit des recommandations** pour optimiser vos campagnes de netlinking
    
    ### 📊 Résultats obtenus :
    
    - Tableau priorisé des domaines à contacter
    - Visualisations interactives
    - Export des résultats en CSV
    - Recommandations automatiques
    - Analyse par segments de priorité
    
    **Commencez par uploader votre export Ahrefs "Referring Domains" dans la barre latérale !**
    """)
    
    # Afficher un exemple de structure attendue
    st.subheader("📁 Structure des fichiers attendus")
    
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
st.markdown("**Développé pour optimiser vos campagnes de netlinking SEO** 🎯")
