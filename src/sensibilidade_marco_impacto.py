# -*- coding: utf-8 -*-
"""
SENSIBILIDADE AO MARCO DE SEGMENTAÇÃO
=====================================

O script principal define o início de cada ciclo pelo pico da aceleração
anteroposterior (AccZ filtrada a 6 Hz). Esse pico ocorre por volta da saída
do pé (~270 ms após o impacto, ~500 ms antes do seguinte), e sua latência varia de
7 a 12 ms (DP) entre passadas. Essa imprecisão do marco entra nas medidas
como se fosse variabilidade do corredor.

Este script repete as medidas usando como marco o pico de impacto da
aceleração longitudinal (AccY a 20 Hz), uma estimativa do contato inicial
e o início convencional do ciclo da passada, e compara as duas versões.

Detecção do impacto (heurística simples, não validada): maior pico de AccY
entre 45% e 80% de cada ciclo definido pelo marco original.

SAÍDAS (na pasta results)
-------------------------
    sensibilidade_marco_impacto.csv
    figura_ciclos_alinhada_impacto.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analise_acelerometria_hurst as base

N_CICLOS = base.N_CICLOS_FORMA_ONDA
JANELA_IMPACTO = (0.45, 0.80)


def normalizar_ciclos(sinal: np.ndarray, marcos: np.ndarray) -> np.ndarray:
    """Interpola cada ciclo entre marcos consecutivos em 101 pontos."""
    destino = np.linspace(0.0, 1.0, base.PONTOS_POR_CICLO)
    ciclos = []
    for inicio, fim in zip(marcos[:-1], marcos[1:]):
        segmento = sinal[inicio : fim + 1]
        origem = np.linspace(0.0, 1.0, len(segmento))
        ciclos.append(np.interp(destino, origem, segmento))
    return np.asarray(ciclos)


def variabilidade_pct(matriz: np.ndarray) -> float:
    """Mesma métrica do script principal: DP médio / amplitude da média."""
    dp = np.std(matriz, axis=0, ddof=1)
    return float(100.0 * np.mean(dp) / np.ptp(matriz.mean(axis=0)))


def cv_central(intervalos: np.ndarray) -> float:
    serie = base.selecionar_trecho_central(intervalos, n=N_CICLOS)
    return float(100.0 * np.std(serie, ddof=1) / np.mean(serie))


def detectar_impactos(acc_y_20: np.ndarray, picos_z: np.ndarray) -> np.ndarray:
    impactos = []
    for inicio, fim in zip(picos_z[:-1], picos_z[1:]):
        duracao = fim - inicio
        j0 = inicio + int(JANELA_IMPACTO[0] * duracao)
        j1 = inicio + int(JANELA_IMPACTO[1] * duracao)
        impactos.append(j0 + int(np.argmax(acc_y_20[j0:j1])))
    return np.asarray(impactos)


def trecho_central(marcos: np.ndarray) -> np.ndarray:
    inicio = (len(marcos) - (N_CICLOS + 1)) // 2
    return marcos[inicio : inicio + N_CICLOS + 1]


def processar() -> tuple[pd.DataFrame, dict]:
    linhas = []
    matrizes = {}
    for caminho in sorted(base.PASTA_DADOS.glob("*.txt")):
        meta = base.interpretar_nome_arquivo(caminho)
        dados = pd.read_csv(caminho, sep="\t", low_memory=False)
        for dispositivo, perna_df in dados.groupby("DeviceName", sort=False):
            perna_df = perna_df.reset_index(drop=True)
            acc_y = perna_df[base.COLUNA_LONGITUDINAL].to_numpy(dtype=float)
            acc_z = perna_df[base.COLUNA_ANTEROPOSTERIOR].to_numpy(dtype=float)
            acc_y_20 = base.filtrar_sinal(acc_y, base.CORTE_FORMA_ONDA_HZ)

            picos_z, intervalos_z = base.detectar_intervalos_passada(acc_z)
            impactos = detectar_impactos(acc_y_20, picos_z)
            latencia = (impactos - picos_z[:-1]) / base.FS

            ciclos_z = normalizar_ciclos(acc_y_20, trecho_central(picos_z))
            ciclos_imp = normalizar_ciclos(acc_y_20, trecho_central(impactos))
            perna = base.identificar_perna(str(dispositivo))
            matrizes[(meta["participante"], meta["condicao"], perna)] = ciclos_imp

            linhas.append(
                {
                    **meta,
                    "perna": perna,
                    "latencia_z_impacto_ms": 1000 * float(np.mean(latencia)),
                    "dp_latencia_ms": 1000 * float(np.std(latencia, ddof=1)),
                    "var_forma_onda_y_pct_marco_z": variabilidade_pct(ciclos_z),
                    "var_forma_onda_y_pct_marco_impacto": variabilidade_pct(ciclos_imp),
                    "cv_passada_pct_marco_z": cv_central(intervalos_z),
                    "cv_passada_pct_marco_impacto": cv_central(np.diff(impactos) / base.FS),
                }
            )
    return pd.DataFrame(linhas), matrizes


def desenhar(matriz: np.ndarray, destino) -> None:
    eixo_x = np.linspace(0, 100, matriz.shape[1])
    figura, eixo = plt.subplots(figsize=(7.4, 4.4), dpi=300)
    for ciclo in matriz:
        eixo.plot(eixo_x, ciclo, color="#C3CFD8", linewidth=0.7, alpha=0.85)
    eixo.plot(eixo_x, matriz.mean(axis=0), color="#0A5C82", linewidth=2.6)
    eixo.set_xlim(0, 100)
    eixo.set_xlabel("Ciclo da passada (%) — 0% = pico de impacto", color="#5B6770")
    eixo.set_ylabel("Aceleração longitudinal (g)", color="#5B6770")
    eixo.tick_params(colors="#5B6770")
    for lado in ["top", "right"]:
        eixo.spines[lado].set_visible(False)
    eixo.grid(axis="y", color="#EDF1F4", linewidth=0.8)
    eixo.set_axisbelow(True)
    figura.tight_layout()
    figura.savefig(destino, bbox_inches="tight", facecolor="white")
    plt.close(figura)


def main() -> None:
    base.PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    tabela, matrizes = processar()
    tabela.to_csv(
        base.PASTA_RESULTADOS / "sensibilidade_marco_impacto.csv",
        index=False, sep=";", decimal=",", encoding="utf-8-sig",
    )
    print(tabela.round(2).to_string(index=False))

    chave = next((k for k in matrizes if k[1] == "PREF" and k[2] == "E"), None)
    if chave:
        desenhar(
            matrizes[chave],
            base.PASTA_RESULTADOS / "figura_ciclos_alinhada_impacto.png",
        )


if __name__ == "__main__":
    main()
