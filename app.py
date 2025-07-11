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
        st.header("Résultats de l'analyse")
        
        # Export des résultats
        st.subheader("Télécharger les résultats")
        
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
