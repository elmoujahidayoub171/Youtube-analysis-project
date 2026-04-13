import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from flask import Flask, render_template

app = Flask(__name__)

# ── DATA (reproduit le dataset du notebook) ────────────────────────────────────
np.random.seed(42)
n = 995

# subscribers (millions) – distribution log-normale asymétrique à droite
subs_raw = np.random.lognormal(mean=3.3, sigma=0.55, size=n)
subs_raw = np.clip(subs_raw, 12.3, 245)
# forcer quelques super-chaînes
subs_raw[:9] = [245, 170, 166, 162, 159, 134, 111, 111, 106]
subs_millions = pd.Series(subs_raw)

# video views (milliards) – log-normale très asymétrique
views_raw = np.random.lognormal(mean=1.5, sigma=1.8, size=n) * 1e9
views_raw = np.clip(views_raw, 0, 2.28e11)
video_views = pd.Series(views_raw)

# uploads – log-normale avec extrêmes TV indiennes
uploads_raw = np.random.lognormal(mean=5.5, sigma=1.8, size=n).astype(int)
uploads_raw = np.clip(uploads_raw, 1, 120000)
uploads_raw[:3] = [116536, 98543, 61705]
uploads = pd.Series(uploads_raw)

# earnings_midpoint
earn_low  = np.random.lognormal(mean=12.5, sigma=1.4, size=n)
earn_high = earn_low * np.random.uniform(2, 18, size=n)
earnings_midpoint = pd.Series((earn_low + earn_high) / 2)

# Country
countries = ["United States"]*303 + ["India"]*176 + ["Brazil"]*67 + \
            ["United Kingdom"]*52 + ["Mexico"]*28 + ["South Korea"]*20 + \
            ["Canada"]*18 + ["Indonesia"]*17 + ["Unknown"]*122 + \
            ["Other"]*(n - 303 - 176 - 67 - 52 - 28 - 20 - 18 - 17 - 122)
np.random.shuffle(countries)
country = pd.Series(countries[:n])

# channel_type
types = (["Entertainment"]*319 + ["Music"]*222 + ["Games"]*96 +
         ["People"]*79 + ["Comedy"]*53 + ["Education"]*52 +
         ["Film"]*39 + ["Howto"]*36 + ["News"]*32 +
         ["Tech"]*17 + ["Sports"]*14 + ["Animals"]*4 +
         ["Nonprofit"]*3 + ["Autos"]*2)
np.random.shuffle(types)
channel_type = pd.Series(types[:n])

df = pd.DataFrame({
    "subscribers":       subs_millions,
    "video views":       video_views,
    "uploads":           uploads,
    "earnings_midpoint": earnings_midpoint,
    "Country":           country,
    "channel_type":      channel_type,
})


# ── HELPER ─────────────────────────────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded

DARK_BG   = "#07090f"
CARD_BG   = "#101520"
TEXT_COL  = "#e8eaf0"
GRID_COL  = "#1e2535"

def base_style(fig, ax_list=None):
    fig.patch.set_facecolor(CARD_BG)
    for ax in (ax_list or fig.get_axes()):
        ax.set_facecolor(CARD_BG)
        ax.tick_params(colors=TEXT_COL, labelsize=10)
        ax.xaxis.label.set_color(TEXT_COL)
        ax.yaxis.label.set_color(TEXT_COL)
        ax.title.set_color(TEXT_COL)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(axis="y", linestyle="--", alpha=0.25, color=GRID_COL)


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPHIQUES DU NOTEBOOK – reproduits exactement
# ═══════════════════════════════════════════════════════════════════════════════

# ── COUNTRY ───────────────────────────────────────────────────────────────────
def chart_country_barplot():
    fig, ax = plt.subplots(figsize=(14, 5))
    base_style(fig, [ax])
    sns.barplot(x=df["Country"].value_counts().index,
                y=df["Country"].value_counts().values,
                ax=ax, color="#4a90d9")
    ax.set_title("Visualisation Country (toutes valeurs)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Country", fontsize=12)
    ax.set_ylabel("Nbr", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_country_top10_bar():
    top10 = df["Country"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    base_style(fig, [ax])
    top10.plot(kind="bar", ax=ax, color="#4a90d9", edgecolor=GRID_COL)
    ax.set_title("Top 10 pays – nombre de chaînes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Pays", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_country_top10_pie():
    top10 = df["Country"].value_counts().head(10)
    colors = ["#e53935","#2979ff","#00bfa5","#ffd740","#ab47bc",
              "#ff7043","#26c6da","#66bb6a","#ec407a","#42a5f5"]
    fig, ax = plt.subplots(figsize=(8, 6))
    base_style(fig, [ax])
    top10.plot(kind="pie", autopct="%1.1f%%", colors=colors,
               startangle=90, ax=ax, textprops={"color": TEXT_COL, "fontsize": 9})
    ax.set_title("Top 10 pays – Diagramme circulaire", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_country_pareto():
    counts = df["Country"].value_counts().head(10)
    cumulative = counts.cumsum() / counts.sum() * 100
    fig, ax1 = plt.subplots(figsize=(10, 5))
    base_style(fig, [ax1])
    ax1.bar(counts.index, counts.values, color="#4a90d9", edgecolor=GRID_COL)
    ax1.set_xticklabels(counts.index, rotation=45, ha="right", fontsize=9, color=TEXT_COL)
    ax1.set_ylabel("Nombre de chaînes", fontsize=11, color=TEXT_COL)
    ax1.set_title("Diagramme de Pareto – Country", fontsize=14, fontweight="bold", color=TEXT_COL)
    ax2 = ax1.twinx()
    ax2.set_facecolor(CARD_BG)
    ax2.plot(range(len(counts)), cumulative.values, color="#e53935",
             marker="o", linewidth=2, markersize=5)
    ax2.set_ylabel("Pourcentage cumulatif (%)", fontsize=11, color=TEXT_COL)
    ax2.tick_params(colors=TEXT_COL)
    ax2.set_ylim(0, 110)
    fig.tight_layout()
    return fig_to_b64(fig)

# ── CHANNEL TYPE ──────────────────────────────────────────────────────────────
def chart_type_bar():
    counts = df["channel_type"].value_counts()
    colors = ["#FF9999","#66B3FF","#99FF99","#FFCC99","#FFD700","#C0C0C0",
              "#ff7043","#26c6da","#ab47bc","#66bb6a","#ec407a","#42a5f5","#ffd740","#26a69a"]
    fig, ax = plt.subplots(figsize=(11, 5))
    base_style(fig, [ax])
    counts.plot(kind="bar", color=colors[:len(counts)], edgecolor="black", ax=ax)
    ax.set_title("Distribution des types de chaînes", fontsize=14, fontweight="bold")
    ax.set_xlabel("Type de chaîne", fontsize=12)
    ax.set_ylabel("Nombre de chaînes", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_type_pie():
    counts = df["channel_type"].value_counts()
    colors = ["#FF9999","#66B3FF","#99FF99","#FFCC99","#FFD700","#C0C0C0",
              "#ff7043","#26c6da","#ab47bc","#66bb6a","#ec407a","#42a5f5","#ffd740","#26a69a"]
    fig, ax = plt.subplots(figsize=(8, 7))
    base_style(fig, [ax])
    counts.plot(kind="pie", autopct="%1.1f%%", colors=colors[:len(counts)],
                startangle=90, shadow=True, ax=ax,
                textprops={"color": TEXT_COL, "fontsize": 8})
    ax.set_title("Distribution of Channel Types", fontsize=14, fontweight="bold")
    ax.set_ylabel("")
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_type_pareto():
    counts = df["channel_type"].value_counts().sort_values(ascending=False)
    cumulative_percent = counts.cumsum() / counts.sum() * 100
    fig, ax1 = plt.subplots(figsize=(10, 5))
    base_style(fig, [ax1])
    counts.plot(kind="bar", color="skyblue", edgecolor="black", ax=ax1)
    ax1.set_ylabel("Nombre de chaînes", fontsize=11, color=TEXT_COL)
    ax1.set_xlabel("Type de chaîne", fontsize=11, color=TEXT_COL)
    ax1.set_title("Diagramme de Pareto – types de chaînes", fontsize=14, fontweight="bold", color=TEXT_COL)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.set_facecolor(CARD_BG)
    ax2.plot(range(len(counts)), cumulative_percent.values,
             color="red", marker="o", linestyle="-", linewidth=2, markersize=5)
    ax2.set_ylabel("Pourcentage cumulatif (%)", fontsize=11, color=TEXT_COL)
    ax2.tick_params(colors=TEXT_COL)
    ax2.set_ylim(0, 110)
    fig.tight_layout()
    return fig_to_b64(fig)

# ── VIDEO VIEWS ───────────────────────────────────────────────────────────────
def chart_views_hist():
    fig, ax = plt.subplots(figsize=(14, 6))
    base_style(fig, [ax])
    sns.histplot(df["video views"], bins=90, edgecolor="black",
                 color="#2fff05", ax=ax)
    ax.set_title("Distribution du nombre de vues", fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre de vues", fontsize=13)
    ax.set_ylabel("Fréquence", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_views_kde():
    fig, ax = plt.subplots(figsize=(14, 6))
    base_style(fig, [ax])
    sns.kdeplot(df["video views"], fill=True, color="#2fff05",
                linewidth=2, alpha=0.5, ax=ax)
    ax.set_title("Distribution du nombre de vues", fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre de vues", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_views_boxplot():
    fig, ax = plt.subplots(figsize=(14, 5))
    base_style(fig, [ax])
    sns.boxplot(x=df["video views"], fill=True, color="#d43131",
                linewidth=1.5, ax=ax)
    ax.set_title("Distribution du nombre de vues – Boîte à moustaches",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre de vues", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_views_violin():
    fig, ax = plt.subplots(figsize=(14, 6))
    base_style(fig, [ax])
    sns.violinplot(x=df["video views"], fill=True, color="#fffb05",
                   linewidth=1.5, ax=ax)
    ax.set_title("Distribution du nombre de vues – Diagramme en violon",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre de vues", fontsize=13)
    ax.set_ylabel("Fréquence", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)

# ── UPLOADS ───────────────────────────────────────────────────────────────────
def chart_uploads_hist():
    fig, ax = plt.subplots(figsize=(14, 6))
    base_style(fig, [ax])
    sns.histplot(df["uploads"], bins=40, edgecolor="black",
                 color="#b005ff", ax=ax)
    ax.set_title("Distribution du nombre de vidéos publiées", fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre de vidéos publiées", fontsize=13)
    ax.set_ylabel("Fréquence", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_uploads_kde():
    fig, ax = plt.subplots(figsize=(14, 6))
    base_style(fig, [ax])
    sns.kdeplot(df["uploads"], fill=True, color="#b005ff",
                linewidth=2, ax=ax)
    ax.set_title("Distribution du nombre de vidéos publiées", fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre de vidéos publiées", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_uploads_boxplot():
    fig, ax = plt.subplots(figsize=(14, 5))
    base_style(fig, [ax])
    sns.boxplot(x=df["uploads"], fill=True, color="#d43131",
                linewidth=1.5, ax=ax)
    ax.set_title("Distribution du nombre de vidéos publiées – Boîte à moustaches",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre de vidéos publiées", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_uploads_violin():
    fig, ax = plt.subplots(figsize=(14, 6))
    base_style(fig, [ax])
    sns.violinplot(x=df["uploads"], fill=True, color="#fffb05",
                   linewidth=1.5, ax=ax)
    ax.set_title("Distribution du nombre de vidéos publiées – Diagramme en violon",
                 fontsize=16, fontweight="bold")
    ax.set_xlabel("Nombre de vidéos publiées", fontsize=13)
    ax.set_ylabel("Fréquence", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)

# ── EARNINGS MIDPOINT ────────────────────────────────────────────────────────
def chart_earn_hist():
    fig, ax = plt.subplots(figsize=(10, 5))
    base_style(fig, [ax])
    sns.histplot(data=df["earnings_midpoint"], stat="count", ax=ax, color="#4a90d9")
    ax.set_title("Distribution of Mid-Earnings for Top YouTubers in 2023",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("earnings_midpoint", fontsize=12)
    ax.set_ylabel("count", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_earn_kde():
    fig, ax = plt.subplots(figsize=(10, 5))
    base_style(fig, [ax])
    sns.kdeplot(df["earnings_midpoint"], fill=True, color="Green", ax=ax)
    ax.set_title("Distribution of Mid-Earnings for Top YouTubers in 2023",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("earnings_midpoint", fontsize=12)
    ax.set_ylabel("Fréquence", fontsize=12)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_earn_violin():
    fig, ax = plt.subplots(figsize=(10, 5))
    base_style(fig, [ax])
    sns.violinplot(data=df, y="earnings_midpoint", color="gold", ax=ax)
    ax.set_title("Distribution of Mid-Earnings for Top YouTubers in 2023",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("earnings_midpoint", fontsize=12)
    sns.despine(ax=ax, left=False, bottom=True)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_earn_boxplot():
    fig, ax = plt.subplots(figsize=(10, 5))
    base_style(fig, [ax])
    sns.boxplot(x=df["earnings_midpoint"], color="skyblue", ax=ax)
    ax.set_title("Boxplot des revenus moyens des top YouTubeurs (2023)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("earnings_midpoint", fontsize=12)
    sns.despine(ax=ax, left=False, bottom=True)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    return fig_to_b64(fig)

# ── SUBSCRIBERS ───────────────────────────────────────────────────────────────
def chart_subs_kde():
    density = gaussian_kde(subs_millions.dropna())
    x_range = np.linspace(0, subs_millions.max(), 500)
    y_density = density(x_range)
    fig, ax = plt.subplots(figsize=(12, 6))
    base_style(fig, [ax])
    ax.plot(x_range, y_density, "r-", linewidth=2.5, label="Courbe de densité (KDE)")
    ax.set_title("Distribution des abonnés YouTube – Courbe de Densité",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Nombre d'abonnés (millions)", fontsize=11)
    ax.set_ylabel("Densité de probabilité", fontsize=11)
    ax.legend(loc="upper right", fontsize=10, facecolor=CARD_BG,
              edgecolor=GRID_COL, labelcolor=TEXT_COL)
    ax.set_xlim(0, 150)
    ax.grid(True, alpha=0.3, color=GRID_COL)
    fig.tight_layout()
    return fig_to_b64(fig)

def chart_subs_dotplot():
    fig, ax = plt.subplots(figsize=(12, 3))
    base_style(fig, [ax])
    y_positions = np.zeros(len(subs_millions))
    ax.scatter(subs_millions, y_positions, alpha=0.6, color="purple", s=20)
    ax.set_yticks([])
    ax.set_xlabel("Nombre d'abonnés (millions)", fontsize=11)
    ax.set_title("Diagramme en points – Chaque valeur individuellement",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 250)
    ax.grid(axis="x", alpha=0.3, color=GRID_COL)
    fig.tight_layout()
    return fig_to_b64(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════
@app.route("/")
def index():
    charts = {
        # Country
        "country_barplot":   chart_country_barplot(),
        "country_top10_bar": chart_country_top10_bar(),
        "country_top10_pie": chart_country_top10_pie(),
        "country_pareto":    chart_country_pareto(),
        # Channel type
        "type_bar":    chart_type_bar(),
        "type_pie":    chart_type_pie(),
        "type_pareto": chart_type_pareto(),
        # Video views
        "views_hist":    chart_views_hist(),
        "views_kde":     chart_views_kde(),
        "views_boxplot": chart_views_boxplot(),
        "views_violin":  chart_views_violin(),
        # Uploads
        "uploads_hist":    chart_uploads_hist(),
        "uploads_kde":     chart_uploads_kde(),
        "uploads_boxplot": chart_uploads_boxplot(),
        "uploads_violin":  chart_uploads_violin(),
        # Earnings
        "earn_hist":    chart_earn_hist(),
        "earn_kde":     chart_earn_kde(),
        "earn_violin":  chart_earn_violin(),
        "earn_boxplot": chart_earn_boxplot(),
        # Subscribers
        "subs_kde":     chart_subs_kde(),
        "subs_dotplot": chart_subs_dotplot(),
    }
    return render_template("dashboard.html", charts=charts)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
