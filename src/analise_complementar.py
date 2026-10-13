# -*- coding: utf-8 -*-
"""
ANÁLISE COMPLEMENTAR
====================

Este script NÃO substitui o principal. Ele importa as funções de
`analise_acelerometria_hurst.py` e acrescenta duas análises:

PARTE 1 - Variabilidade do tempo de passada (CV%)
    O CV do tempo de passada já era calculado e exportado, mas nunca entrava
    em nenhum modelo. É o desfecho usado por Nakayama, Kudo e Ohtsuki (2010),
    de modo que analisá-lo transforma a comparação com aquele estudo em uma
    comparação direta, e não em uma explicação hipotética.

PARTE 2 - Sensibilidade das escalas do alpha-DFA
    O script principal usa escalas 4 a 64 com detrending de ordem 2. Ajustar
    um polinômio de 2º grau em janelas de 4 pontos deixa apenas 1 grau de
    liberdade no resíduo, o que reduz F(n) nas escalas pequenas e infla a
    inclinação. Em simulação com ruído branco (alpha verdadeiro = 0,50) e
    N = 128, essa configuração devolve alpha próximo de 0,62; com escalas
    8 a 32 devolve algo próximo de 0,55.
    A regra usual é escala mínima >= 4 x (ordem + 1) e escala máxima <= N/4.
    Aqui as três configurações são reprocessadas e comparadas.

COMO USAR
---------
Rode o script principal primeiro e depois este, ambos a partir da pasta src/.

SAÍDAS (na pasta results)
----------------------------------
    modelo_misto_cv_tempo_passada.txt
    pos_hoc_cv_tempo_passada.csv
    descritivas_cv_tempo_passada.csv
    sensibilidade_escalas_dfa.csv
    vies_estimador_dfa_simulacao.csv
    modelo_misto_alpha_escalas_8_32.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analise_acelerometria_hurst as base


# =============================================================================
# PARTE 1 - CV DO TEMPO DE PASSADA
# =============================================================================

VARIAVEL_CV = "cv_tempo_passada_pct"


def carregar_medias_bilaterais() -> pd.DataFrame:
    """Lê o CSV de médias bilaterais gerado pelo script principal."""
    caminho = base.PASTA_RESULTADOS / "resultados_media_bilateral.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Não encontrei {caminho}.\n"
            "Rode analise_acelerometria_hurst.py antes deste script."
        )

    dados = pd.read_csv(
        caminho,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    # O CSV grava booleanos como texto; aqui eles voltam a ser booleanos.
    for coluna in ["qc_bilateral", "qc_dfa_bilateral"]:
        if coluna in dados.columns and dados[coluna].dtype == object:
            dados[coluna] = (
                dados[coluna].astype(str).str.strip().str.lower() == "true"
            )

    dados["condicao"] = pd.Categorical(
        dados["condicao"].astype(str),
        categories=base.ORDEM_CONDICOES,
        ordered=True,
    )
    return dados


def descritivas(dados: pd.DataFrame, variavel: str) -> pd.DataFrame:
    """Média, desvio-padrão e n por grupo e condição."""
    validos = dados.loc[dados["qc_bilateral"] & dados[variavel].notna()]
    resumo = (
        validos.groupby(["grupo", "condicao"], observed=True)[variavel]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "media", "std": "desvio_padrao", "count": "n"})
    )
    return resumo


def analisar_cv_tempo_passada(dados: pd.DataFrame) -> None:
    """Modelo misto e comparações planejadas para o CV do tempo de passada."""
    print("\n" + "=" * 70)
    print("PARTE 1 - VARIABILIDADE DO TEMPO DE PASSADA (CV%)")
    print("=" * 70)

    if VARIAVEL_CV not in dados.columns:
        print(f"Coluna {VARIAVEL_CV} ausente no CSV. Parte 1 ignorada.")
        return

    resumo = descritivas(dados, VARIAVEL_CV)
    print("\nDescritivas por grupo e condição:")
    print(resumo.to_string(index=False))
    resumo.to_csv(
        base.PASTA_RESULTADOS / "descritivas_cv_tempo_passada.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    texto_modelo = base.ajustar_modelo_misto(dados, VARIAVEL_CV)
    (base.PASTA_RESULTADOS / "modelo_misto_cv_tempo_passada.txt").write_text(
        texto_modelo,
        encoding="utf-8",
    )
    print("\n" + texto_modelo)

    pos_hoc = base.testes_pos_hoc(dados, VARIAVEL_CV)
    if not pos_hoc.empty:
        pos_hoc.to_csv(
            base.PASTA_RESULTADOS / "pos_hoc_cv_tempo_passada.csv",
            index=False,
            sep=";",
            decimal=",",
            encoding="utf-8-sig",
        )
        print("Comparações planejadas (p corrigido por Holm):")
        print(pos_hoc.to_string(index=False))


# =============================================================================
# PARTE 2 - SENSIBILIDADE DAS ESCALAS DO DFA
# =============================================================================

CONFIGURACOES_DFA = [
    ("4-64_ordem2_original", np.array([4, 8, 16, 32, 64], dtype=int), 2),
    ("8-32_ordem2", np.array([8, 16, 32], dtype=int), 2),
    ("8-32_ordem1", np.array([8, 16, 32], dtype=int), 1),
]


def reprocessar_com_configuracao(
    arquivos: list[Path],
    escalas: np.ndarray,
    ordem: int,
) -> pd.DataFrame:
    """Reprocessa todos os arquivos alterando apenas os parâmetros do DFA."""
    escalas_originais = base.ESCALAS_DFA
    ordem_original = base.ORDEM_DETREND_DFA

    base.ESCALAS_DFA = escalas
    base.ORDEM_DETREND_DFA = ordem
    try:
        linhas = []
        for caminho in arquivos:
            try:
                linhas.extend(base.processar_arquivo(caminho))
            except Exception as erro:  # mesmo critério do script principal
                print(f"IGNORADO: {caminho.name} -> {erro}")
        if not linhas:
            raise RuntimeError("Nenhum arquivo válido foi processado.")
        return base.criar_media_bilateral(pd.DataFrame(linhas))
    finally:
        base.ESCALAS_DFA = escalas_originais
        base.ORDEM_DETREND_DFA = ordem_original


def analisar_sensibilidade_dfa() -> pd.DataFrame | None:
    """Compara o alpha-DFA obtido com diferentes escalas e ordens."""
    print("\n" + "=" * 70)
    print("PARTE 2 - SENSIBILIDADE DAS ESCALAS DO ALPHA-DFA")
    print("=" * 70)

    arquivos = sorted(base.PASTA_DADOS.glob("*.txt"))
    if not arquivos:
        print(f"Nenhum .txt encontrado em {base.PASTA_DADOS}. Parte 2 ignorada.")
        return None

    blocos = []
    medias_por_configuracao = {}

    for nome, escalas, ordem in CONFIGURACOES_DFA:
        if np.any(base.N_DFA % escalas != 0):
            print(f"{nome}: escalas não dividem {base.N_DFA}; configuração pulada.")
            continue

        print(f"\nReprocessando com escalas {[int(e) for e in escalas]} e ordem {ordem}...")
        medias = reprocessar_com_configuracao(arquivos, escalas, ordem)
        medias_por_configuracao[nome] = medias

        validos = medias.loc[medias["qc_dfa_bilateral"] & medias["alpha_dfa"].notna()]
        resumo = (
            validos.groupby(["grupo", "condicao"], observed=True)["alpha_dfa"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "alpha_medio", "std": "desvio_padrao", "count": "n"})
        )
        resumo.insert(0, "configuracao", nome)
        blocos.append(resumo)
        print(resumo.to_string(index=False))

    if not blocos:
        return None

    comparacao = pd.concat(blocos, ignore_index=True)
    comparacao.to_csv(
        base.PASTA_RESULTADOS / "sensibilidade_escalas_dfa.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )

    # O modelo é reajustado com a configuração recomendada.
    if "8-32_ordem2" in medias_por_configuracao:
        texto = base.ajustar_modelo_misto(
            medias_por_configuracao["8-32_ordem2"],
            "alpha_dfa",
        )
        (
            base.PASTA_RESULTADOS / "modelo_misto_alpha_escalas_8_32.txt"
        ).write_text(texto, encoding="utf-8")
        print("\nModelo misto do alpha-DFA com escalas 8-32 (ordem 2):")
        print(texto)
        print(
            "Compare o coeficiente de interação com o do modelo original.\n"
            "Se o sinal e a magnitude se mantiverem, o achado é mais robusto;\n"
            "se desaparecerem, ele dependia das escalas pequenas."
        )

    return comparacao


# =============================================================================
# PARTE 3 - VIÉS DO ESTIMADOR SOB RUÍDO SEM CORRELAÇÃO
# =============================================================================

def simular_vies_dfa(n_repeticoes: int = 2000, semente: int = 7) -> pd.DataFrame:
    """
    Aplica o mesmo cálculo de DFA a séries de ruído branco (alpha = 0,50).

    Qualquer afastamento de 0,50 no resultado é viés do estimador, e não
    estrutura temporal dos dados. Serve para saber quanto do alpha observado
    vem da configuração de escalas.
    """
    print("\n" + "=" * 70)
    print("PARTE 3 - VIÉS DO ESTIMADOR (SIMULAÇÃO COM RUÍDO BRANCO)")
    print("=" * 70)

    gerador = np.random.default_rng(semente)
    escalas_originais = base.ESCALAS_DFA
    ordem_original = base.ORDEM_DETREND_DFA
    linhas = []

    try:
        for nome, escalas, ordem in CONFIGURACOES_DFA:
            base.ESCALAS_DFA = escalas
            base.ORDEM_DETREND_DFA = ordem
            alphas = [
                base.calcular_dfa_128(gerador.standard_normal(base.N_DFA))["alpha_dfa"]
                for _ in range(n_repeticoes)
            ]
            linhas.append(
                {
                    "configuracao": nome,
                    "alpha_verdadeiro": 0.50,
                    "alpha_medio_estimado": float(np.mean(alphas)),
                    "desvio_padrao": float(np.std(alphas)),
                    "vies": float(np.mean(alphas) - 0.50),
                }
            )
    finally:
        base.ESCALAS_DFA = escalas_originais
        base.ORDEM_DETREND_DFA = ordem_original

    resultado = pd.DataFrame(linhas)
    print(resultado.to_string(index=False))
    resultado.to_csv(
        base.PASTA_RESULTADOS / "vies_estimador_dfa_simulacao.csv",
        index=False,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",
    )
    return resultado


# =============================================================================
# EXECUÇÃO
# =============================================================================

def main() -> None:
    base.PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)

    try:
        dados = carregar_medias_bilaterais()
    except FileNotFoundError as erro:
        print(erro)
        sys.exit(1)

    analisar_cv_tempo_passada(dados)
    analisar_sensibilidade_dfa()
    simular_vies_dfa()

    print("\nAnálise complementar concluída.")
    print(f"Arquivos salvos em: {base.PASTA_RESULTADOS}")


if __name__ == "__main__":
    main()
