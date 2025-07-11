# Audit de netlinking - Création d'une roadmap pour satelliser ta stratégie

Un outil Streamlit avancé pour analyser et prioriser vos opportunités de netlinking en croisant les données Ahrefs et Google Search Console.

## Objectif

Cet outil vous permet de :
- Identifier les domaines prioritaires pour vos campagnes de netlinking
- Analyser les gaps concurrentiels (sites qui lient à vos concurrents mais pas à vous)
- Calculer un score de priorité intelligent basé sur multiple critères
- Optimiser votre budget netlinking en se concentrant sur les opportunités les plus rentables

## Installation

```bash
git clone https://github.com/votre-username/netlinking-analysis-tool
cd netlinking-analysis-tool
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Fichiers requis

### 1. Export Ahrefs - Referring Domains (obligatoire)
- Outil : Link Intersect d'Ahrefs
- Format : CSV avec tabulations
- Colonnes : Domain, Domain rating, Domain traffic, Intersect + colonnes par concurrent

### 2. Export Ahrefs - Referring Pages (optionnel)
- Même source que Referring Domains
- Permet une analyse plus granulaire au niveau des pages

### 3. Export Google Search Console - Requêtes (optionnel)
- Performance > Requêtes > Export
- Colonnes : Requêtes, Clics, Impressions, CTR, Position

### 4. Export Google Search Console - Pages (optionnel)
- Performance > Pages > Export  
- Colonnes : Pages, Clics, Impressions, CTR, Position

### 5. Mots-clés stratégiques (optionnel)
- Format : Excel (.xlsx, .xls) ou CSV
- Colonnes suggérées : Keyword, Search Volume, Keyword Difficulty, CPC

### 6. Export SERPs (optionnel)
- Données des top 10 pour vos mots-clés stratégiques
- Format : Excel (.xlsx, .xls) ou CSV

## Algorithme de scoring

Le score de priorité est calculé selon cette formule :

**Score = (DR × 0.2) + (Traffic × 0.2) + (Gap × 0.3) + (Pertinence × 0.3)**

Où :
- **DR** : Domain Rating Ahrefs (normalisé sur 100)
- **Traffic** : Trafic mensuel du domaine (normalisé sur 100) 
- **Gap** : Pourcentage de concurrents qui reçoivent des liens de ce domaine
- **Pertinence** : Score thématique basé sur vos mots-clés et pages performantes

## Fonctionnalités

### Analyse et filtrage
- Filtrage par Domain Rating minimum
- Filtrage par trafic minimum  
- Filtrage par nombre de concurrents ayant des liens
- Calcul automatique des scores de priorité

### Visualisations
- Graphique scatter plot (DR vs Trafic vs Score)
- Graphique en barres des top domaines
- Répartition par segments de priorité
- Métriques clés en temps réel

### Export et recommandations
- Export CSV des résultats complets
- Export des Top N domaines
- Recommandations automatiques Top 3
- Analyse des gaps concurrentiels
- Segmentation par niveau de priorité

## Paramètres configurables

- **Domain Rating minimum** : Filtre les domaines sous un certain DR
- **Trafic minimum** : Exclut les domaines avec peu de trafic
- **Concurrents minimum** : Ne garde que les domaines liés à X concurrents minimum

## Cas d'usage

### Pour une agence SEO
- Prioriser les prospects de netlinking pour vos clients
- Démontrer les opportunités manquées face à la concurrence
- Optimiser le ROI des campagnes de netlinking

### Pour un SEO in-house  
- Identifier rapidement les meilleurs domaines à contacter
- Analyser la stratégie netlinking des concurrents
- Suivre l'évolution des opportunités dans le temps

## Personnalisation

L'algorithme de scoring peut être facilement modifié dans la fonction `calculate_priority_score()` pour s'adapter à vos critères spécifiques.

## Support

Pour toute question ou bug, ouvrez une issue sur GitHub.

## Licence

MIT License - Libre d'utilisation et de modification.

---

**Développé par [JC Espinosa](https://jc-espinosa.com) pour optimiser vos campagnes de netlinking SEO**
