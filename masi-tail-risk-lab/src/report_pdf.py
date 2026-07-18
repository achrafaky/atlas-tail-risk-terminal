"""
MASI Tail-Risk Lab — Rapport PDF de risque (one-click)
=======================================================
Génère un rapport de risque synthétique de 2 pages (style note interne de
desk) avec les KPI, les paramètres EVT, la comparaison des marges et le
verdict de backtesting. Utilise reportlab (rendu déterministe, sans LaTeX).
"""
from __future__ import annotations

import io
from datetime import datetime


def build_risk_report(kpis: dict, evt_params: dict, margins: dict,
                      backtest: dict | None = None) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer,
                                    Table, TableStyle)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.8 * cm,
                            bottomMargin=1.8 * cm)
    ss = getSampleStyleSheet()
    NAVY = colors.HexColor("#1B4F9B")
    title = ParagraphStyle("t", parent=ss["Title"], textColor=NAVY, fontSize=18)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=NAVY)
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=10, leading=14)

    def kv_table(rows):
        t = Table(rows, colWidths=[8 * cm, 6 * cm], hAlign="LEFT")
        t.setStyle(TableStyle([
            ("FONT", (0, 0), (-1, -1), "Helvetica", 10),
            ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 10),
            ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.HexColor("#F2F5FA")]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LINEBELOW", (0, -1), (-1, -1), 0.5, NAVY)]))
        return t

    story = [
        Paragraph("MASI 20 — Rapport de Risque Extrême (EVT)", title),
        Paragraph(f"Généré le {datetime.now():%d/%m/%Y %H:%M} · "
                  "Achraf Akiyaf — Master MMSD, FST Tanger", ss["Italic"]),
        Spacer(1, 14),
        Paragraph("1. Indicateurs clés", h2),
        kv_table([
            ["Niveau du MASI", f"{kpis.get('spot', 0):,.0f} points"],
            ["Volatilité (GARCH, annualisée)", f"{kpis.get('vol_ann', 0):.1f} %"],
            ["VaR 99 % EVT (1 jour)", f"−{kpis.get('var_evt', 0):.2f} %"],
            ["Expected Shortfall 99 %", f"−{kpis.get('es_evt', 0):.2f} %"],
            ["Indice de queue ξ (GPD)", f"{kpis.get('xi', 0):.3f}"],
            ["P(perte > 5 % sous 30 j)", f"{kpis.get('p_crisis', 0):.1f} %"],
        ]),
        Spacer(1, 12),
        Paragraph("2. Paramètres du modèle EVT", h2),
        kv_table([
            ["Seuil POT (u)", f"{evt_params.get('threshold', 0):.2f} %"],
            ["Nombre d'excès", f"{evt_params.get('n_exceed', 0)}"],
            ["ξ (GEV bloc-maxima)", f"{evt_params.get('xi_gev', 0):.3f}"],
            ["ξ (GPD, POT)", f"{evt_params.get('xi_gpd', 0):.3f}"],
            ["Famille de queue", evt_params.get("family", "Fréchet")],
        ]),
        Spacer(1, 12),
        Paragraph("3. Marge initiale du Future MASI 20 (MPOR 2j, 99.5 %)", h2),
        kv_table([
            ["Marge gaussienne", f"{margins.get('gauss', 0):,.0f} MAD"],
            ["Marge historique", f"{margins.get('hist', 0):,.0f} MAD"],
            ["Marge EVT (inconditionnelle)", f"{margins.get('evt', 0):,.0f} MAD"],
            ["Marge GARCH-EVT (conditionnelle)", f"{margins.get('garch', 0):,.0f} MAD"],
            ["Dépôt de garantie de référence", "1 000 MAD"],
        ]),
    ]
    if backtest:
        story += [Spacer(1, 12),
                  Paragraph("4. Backtesting réglementaire", h2),
                  kv_table([
                      ["Verdict", f"{backtest.get('color','')} {backtest.get('verdict','')}"],
                      ["Violations observées / attendues",
                       f"{backtest.get('obs','?')} / {backtest.get('expected','?')}"],
                      ["Test de Kupiec (p-value)", f"{backtest.get('kupiec_p','?')}"],
                      ["Test de Christoffersen (p-value)", f"{backtest.get('christoffersen_p','?')}"],
                  ])]
    story += [Spacer(1, 16),
              Paragraph("<i>Projet académique — Master MMSD. Indicateurs "
                        "statistiques de régime de risque, ne constituant pas "
                        "un conseil en investissement ni un audit de la CCP.</i>",
                        body)]
    doc.build(story)
    return buf.getvalue()
