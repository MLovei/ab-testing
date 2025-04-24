# A/B/C Testing in Action: Insights from Marketing and Gaming Campaigns

## Table of Contents
- [Introduction](#introduction)
- [Project Aim](#project-aim)
- [Technologies](#technologies)
- [Scope of Functionalities](#scope-of-functionalities)
- [Examples of Use](#examples-of-use)
- [Installation](#installation)
- [Results](#results)
- [Sources](#sources)

## introduction
This analysis delves into the application of A/B/C testing in two distinct scenarios: a marketing campaign with three competing promotions and a mobile game experiment focused on a gate change.

## project-aim
To evaluate the effectiveness of different promotional strategies and game design decisions through rigorous statistical testing, providing actionable insights for marketing teams and game developers.

## technologies
- Python (primary analysis language)
- Jupyter Notebooks
- Statistical libraries (NumPy, SciPy, Pandas, Statsmodels)
- Visualization tools (Matplotlib, Seaborn)
- Bootstrapping methods
- Non-parametric statistical tests

## scope-of-functionalities
### Marketing Campaign Analysis
Three promotions were assessed for their effectiveness using various statistical tests. Initial data exploration revealed non-normal distributions, leading to the use of non-parametric methods such as the Kruskal-Wallis and Mann-Whitney U tests. The results indicated a significant difference in performance, with **Promotion 3** surpassing both **Promotion 2** and **Promotion 1**. Further analysis is necessary to understand the underlying factors contributing to these differences.

### Mobile Game Experiment
We investigated the impact of moving a gate within the game from level 30 to level 40 on player retention. The Mann-Whitney U test did not reveal a statistically significant difference in the number of game rounds played between the two groups. Bootstrapping and analytical methods were employed to estimate retention rates, with **gate_30** showing a slight but inconclusive advantage. Additional data collection and segmentation may be required to uncover more subtle patterns.

## examples-of-use
- Marketing teams can apply similar methodologies to test campaign effectiveness before full-scale deployment
- Game developers can utilize these testing approaches when considering level design changes
- Product managers can implement A/B/C testing to evaluate feature modifications
- Data scientists can reference the statistical methods for analyzing non-normally distributed data

## installation
Refer to `requirements.txt` for a complete list of dependencies. To set up the project environment:

```
git clone [repository-url]
cd [repository-name]
pip install -r requirements.txt
```

## results
### Conclusion
This analysis underscores the importance of A/B/C testing in evaluating marketing strategies and game design decisions. However, it also emphasizes the significance of meticulous experimental design, appropriate statistical methods, and nuanced interpretation of results. Future work should concentrate on expanding data collection efforts, exploring potential confounding variables, and refining experimental designs to gain deeper insights into the factors influencing consumer behavior and player engagement.

## sources
The Jupyter Notebooks in the [src/](src/) folder contain the detailed analysis, code, and visualizations for this project.

