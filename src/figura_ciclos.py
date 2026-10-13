# -*- coding: utf-8 -*-
"""
FIGURA DOS CICLOS SOBREPOSTOS
=============================

Gera, a partir dos seus dados reais, as 100 passadas normalizadas de uma
perna, em cinza, com a curva média por cima, em azul. Serve tanto para a
aceleração longitudinal (desfecho principal) quanto para a resultante
tridimensional (análise de sensibilidade).

A segmentação reproduz exatamente o que o script principal faz — mesmo filtro
de 6 Hz no eixo Z para detectar, mesmo filtro de 20 Hz na forma de onda,
mesmos 100 ciclos centrais interpolados em 101 pontos. Para a resultante, os
três eixos são filtrados antes da raiz quadrada, na mesma ordem do original.
A figura, portanto, não é uma ilustração: é o seu sinal.

COMO USAR
---------
Execute a partir da pasta src/ (os dados são lidos de data/). Sem nenhuma alteração, ele escolhe automaticamente a
perna com melhor qualidade (maior correlação média entre ciclos) na condição
preferida. Para escolher um participante específico, edite PARTICIPANTE abaixo.

SAÍDAS (na pasta results)
----------------------------------
    figura_ciclos_longitudinal.png   — 300 dpi
    figura_ciclos_resultante.png     — 300 dpi
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analise_acelerometria_hurst as base


# ---------------------------------------------------------------------------
# ESCOLHA O QUE PLOTAR
# ---------------------------------------------------------------------------

PARTICIPANTE = None      # None = escolhe automaticamente o de melhor qualidade
CONDICAO = "PREF"        # "-10", "PREF" ou "+10"
PERNA = None             # None, "D" (direita) ou "E" (esquerda)

# Cada sinal vira uma figura. Comente uma linha para gerar apenas a outra.
SINAIS = [
    ("longitudinal", "Aceleração longitudinal (g)"),
    ("resultante", "Aceleração resultante (g)"),
]

COR_CICLOS = "#C3CFD8"   # cinza-azulado dos slides
COR_MEDIA = "#0A5C82"    # azul dos slides
COR_TEXTO = "#5B6770"
FUNDO_TRANSPARENTE = False  # True para usar em slides


def extrair_ciclos(dados_perna: pd.DataFrame, sinal_escolhido: str) -> np.ndarray:
    """
    Devolve a matriz de ciclos normalizados (100 x 101).

    Reproduz passo a passo a segmentação de calcular_variabilidade_forma_onda,
    que calcula a variabilidade mas não devolve os ciclos. Os picos vêm sempre
    do eixo Z, independentemente do sinal plotado — é o mesmo marco de ciclo
    usado no processamento original.
    """
    acc_x = dados_perna[base.COLUNA_MEDIO_LATERAL].to_numpy(dtype=float)
    acc_y = dados_perna[base.COLUNA_LONGITUDINAL].to_numpy(dtype=float)
    acc_z = dados_perna[base.COLUNA_ANTEROPOSTERIOR].to_numpy(dtype=float)

    indices_picos, _ = base.detectar_intervalos_passada(acc_z)
    if len(indices_picos) < base.N_CICLOS_FORMA_ONDA + 1:
        raise ValueError("Picos insuficientes para 100 ciclos.")

    if sinal_escolhido == "resultante":
        # Os três eixos são filtrados antes da raiz quadrada, como no original.
        sinal = np.sqrt(
            base.filtrar_sinal(acc_x, base.CORTE_FORMA_ONDA_HZ) ** 2
            + base.filtrar_sinal(acc_y, base.CORTE_FORMA_ONDA_HZ) ** 2
            + base.filtrar_sinal(acc_z, base.CORTE_FORMA_ONDA_HZ) ** 2
        )
    else:
        sinal = base.filtrar_sinal(acc_y, base.CORTE_FORMA_ONDA_HZ)

    inicio = (len(indices_picos) - (base.N_CICLOS_FORMA_ONDA + 1)) // 2
    picos = indices_picos[inicio : inicio + base.N_CICLOS_FORMA_ONDA + 1]

    destino = np.linspace(0.0, 1.0, base.PONTOS_POR_CICLO)
    ciclos = []
    for p0, p1 in zip(picos[:-1], picos[1:]):
        segmento = sinal[p0 : p1 + 1]
        if len(segmento) < 4:
            raise ValueError("Ciclo curto demais para interpolação.")
        origem = np.linspace(0.0, 1.0, len(segmento))
        ciclos.append(np.interp(destino, origem, segmento))

    return np.asarray(ciclos, dtype=float)


def candidatas(arquivos: list[Path], sinal_escolhido: str) -> list[dict]:
    """Percorre os arquivos da condição escolhida e coleta as pernas válidas."""
    encontradas = []
    for caminho in arquivos:
        try:
            meta = base.interpretar_nome_arquivo(caminho)
        except Exception:
            continue
        if meta["condicao"] != CONDICAO:
            continue
        if PARTICIPANTE and meta["participante"] != PARTICIPANTE:
            continue

        try:
            dados = pd.read_csv(caminho, sep="\t", low_memory=False)
        except Exception as erro:
            print(f"IGNORADO: {caminho.name} -> {erro}")
            continue

        for nome_dispositivo, dados_perna in dados.groupby("DeviceName", sort=False):
            perna = base.identificar_perna(str(nome_dispositivo))
            if PERNA and perna != PERNA:
                continue
            try:
                matriz = extrair_ciclos(
                    dados_perna.reset_index(drop=True), sinal_escolhido
                )
            except Exception as erro:
                print(f"IGNORADO: {caminho.name} / {perna} -> {erro}")
                continue

            correlacoes = np.corrcoef(matriz)
            qualidade = float(
                np.nanmean(correlacoes[np.triu_indices_from(correlacoes, k=1)])
            )
            encontradas.append(
                {
                    "participante": meta["participante"],
                    "grupo": base.GRUPOS.get(meta["participante"], "Não informado"),
                    "perna": perna,
                    "matriz": matriz,
                    "qualidade": qualidade,
                }
            )
    return encontradas


def desenhar(escolhida: dict, destino: Path, rotulo_y: str) -> None:
    matriz = escolhida["matriz"]
    media = matriz.mean(axis=0)
    eixo_x = np.linspace(0, 100, matriz.shape[1])

    figura, eixo = plt.subplots(figsize=(7.4, 4.4), dpi=300)
    if FUNDO_TRANSPARENTE:
        figura.patch.set_alpha(0.0)
        eixo.patch.set_alpha(0.0)

    for ciclo in matriz:
        eixo.plot(eixo_x, ciclo, color=COR_CICLOS, linewidth=0.7, alpha=0.85)
    eixo.plot(eixo_x, media, color=COR_MEDIA, linewidth=2.6)

    eixo.set_xlabel("Ciclo da passada (%)", fontsize=11, color=COR_TEXTO)
    eixo.set_ylabel(rotulo_y, fontsize=11, color=COR_TEXTO)
    eixo.set_xlim(0, 100)
    eixo.tick_params(colors=COR_TEXTO, labelsize=10)
    for lado in ["top", "right"]:
        eixo.spines[lado].set_visible(False)
    for lado in ["left", "bottom"]:
        eixo.spines[lado].set_color("#D5DBDF")
    eixo.grid(axis="y", color="#EDF1F4", linewidth=0.8)
    eixo.set_axisbelow(True)

    figura.tight_layout()
    figura.savefig(
        destino,
        transparent=FUNDO_TRANSPARENTE,
        bbox_inches="tight",
        facecolor="none" if FUNDO_TRANSPARENTE else "white",
    )
    plt.close(figura)


def main() -> None:
    base.PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(base.PASTA_DADOS.glob("*.txt"))
    if not arquivos:
        print(f"Nenhum .txt encontrado em {base.PASTA_DADOS}.")
        return

    referencia = None
    for sinal_escolhido, rotulo_y in SINAIS:
        opcoes = candidatas(arquivos, sinal_escolhido)
        if not opcoes:
            print(f"Nenhuma perna válida para {sinal_escolhido}.")
            continue

        # A primeira figura define participante e perna; a segunda usa os
        # mesmos, para que as duas mostrem exatamente o mesmo trecho de sinal.
        if referencia is None:
            escolhida = max(opcoes, key=lambda d: d["qualidade"])
            referencia = (escolhida["participante"], escolhida["perna"])
        else:
            iguais = [
                o for o in opcoes
                if (o["participante"], o["perna"]) == referencia
            ]
            escolhida = iguais[0] if iguais else max(
                opcoes, key=lambda d: d["qualidade"]
            )

        destino = base.PASTA_RESULTADOS / f"figura_ciclos_{sinal_escolhido}.png"
        desenhar(escolhida, destino, rotulo_y)

        print(
            f"Figura gerada: {destino}\n"
            f"  sinal: {sinal_escolhido}\n"
            f"  participante: {escolhida['participante']} ({escolhida['grupo']})\n"
            f"  condição: {CONDICAO} | perna: {escolhida['perna']}\n"
            f"  ciclos: {escolhida['matriz'].shape[0]} | "
            f"correlação média entre ciclos: {escolhida['qualidade']:.3f}\n"
        )

    print(
        "Nas legendas, descreva assim: 100 passadas consecutivas de um\n"
        "participante na velocidade preferida, com a curva média em azul."
    )


if __name__ == "__main__":
    main()
