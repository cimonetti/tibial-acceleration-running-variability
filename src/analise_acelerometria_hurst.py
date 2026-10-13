"""
Análise de corrida em esteira com acelerômetros tibiais bilaterais.

Pipeline:
1. Lê automaticamente arquivos participante_condição_velocidade.txt.
2. Separa os dois sensores/pernas.
3. Filtra AccZ (anteroposterior) e detecta um máximo por ciclo da mesma perna.
4. Constrói a série de intervalos de passada.
5. Seleciona 100 ciclos centrais para a análise principal.
6. Calcula a variabilidade da forma de onda longitudinal (AccY) e o CV.
7. Calcula a variabilidade da resultante 3D como análise de sensibilidade.
8. Quando disponíveis, usa 128 intervalos para a alpha-DFA exploratória.
9. Calcula a média bilateral e ajusta modelos mistos grupo x condição.

Exemplos de nomes:
    P01_-10_7@2.txt
    P01_PREF_8.txt
    P01_+10_8@8.txt

O caractere @ é interpretado como separador decimal.

Dependências:
    numpy, pandas, scipy e matplotlib

Opcional para o modelo misto:
    statsmodels
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, sosfiltfilt
from scipy.stats import linregress, ttest_ind, ttest_rel

try:
    from statsmodels.formula.api import mixedlm

    STATSMODELS_DISPONIVEL = True
except ImportError:
    mixedlm = None
    STATSMODELS_DISPONIVEL = False


# =============================================================================
# CONFIGURAÇÕES QUE VOCÊ PODE ALTERAR
# =============================================================================

# Por padrão, procura os .txt na pasta data/ do repositório e grava as saídas
# em results/. Também é possível informar outra pasta explicitamente.
try:
    RAIZ_PROJETO = Path(__file__).resolve().parent.parent
except NameError:  # Permite executar células isoladas no Spyder.
    RAIZ_PROJETO = Path.cwd()

PASTA_DADOS = RAIZ_PROJETO / "data"
PASTA_RESULTADOS = RAIZ_PROJETO / "results"

# Participantes identificados por código. A chave deve ser igual ao código
# antes do primeiro separador de condição no nome do arquivo.
# Apenas os dados de P01 são públicos (pasta data/).
GRUPOS = {
    "P01": "Experiente",
    "P02": "Experiente",
    "P03": "Experiente",
    "P04": "Experiente",
    "P05": "Experiente",
    "P06": "Inexperiente",
    "P07": "Inexperiente",
    "P08": "Inexperiente",
    "P09": "Inexperiente",
    "P10": "Inexperiente",
}

# Configurações da aquisição e do processamento.
FS = 200.0                  # Frequência nominal de amostragem em Hz.
ORDEM_FILTRO = 4
CORTE_HZ = 6.0              # Passa-baixa usado apenas para marcar os ciclos.
CORTE_FORMA_ONDA_HZ = 20.0  # Preserva a forma global da aceleração tibial.
PONTOS_POR_CICLO = 101      # Normalização de 0 a 100% da passada.
DISTANCIA_MIN_S = 0.55      # Distância mínima entre máximos do eixo Z.
PROEMINENCIA_MIN_G = 0.50
INTERVALO_MIN_S = 0.55      # Faixa esperada do intervalo de passada.
INTERVALO_MAX_S = 1.10

# Análise principal com 100 ciclos; DFA exploratória preservada com N=128.
N_CICLOS_FORMA_ONDA = 100
N_DFA = 128
ESCALAS_DFA = np.array([4, 8, 16, 32, 64], dtype=int)
ORDEM_DETREND_DFA = 2       # Tendência quadrática.

# Identificação dos sensores. O sensor terminado em D5 é a perna direita.
SUFIXO_DIREITA = "D5"
SUFIXO_ESQUERDA = "90C"

# Orientação confirmada durante a instalação dos sensores:
# X = médio-lateral; Y = longitudinal/axial da tíbia; Z = anteroposterior.
COLUNA_MEDIO_LATERAL = "AccX(g)"
COLUNA_LONGITUDINAL = "AccY(g)"
COLUNA_ANTEROPOSTERIOR = "AccZ(g)"

ORDEM_CONDICOES = ["-10", "PREF", "+10"]
CORES_GRUPOS = {
    "Experiente": "#1f77b4",
    "Inexperiente": "#d62728",
}


# =============================================================================
# FUNÇÕES DE LEITURA E PROCESSAMENTO
# =============================================================================

def interpretar_nome_arquivo(caminho: Path) -> dict:
    """Extrai participante, condição e velocidade do nome do arquivo."""
    partes = caminho.stem.rsplit("_", 2)
    if len(partes) != 3:
        raise ValueError(
            "Nome fora do padrão participante_condição_velocidade.txt"
        )

    participante, condicao, velocidade_txt = partes
    condicao = condicao.upper()
    if condicao not in ORDEM_CONDICOES:
        raise ValueError(f"Condição desconhecida: {condicao}")

    velocidade_txt = velocidade_txt.replace("@", ".").replace(",", ".")
    try:
        velocidade = float(velocidade_txt)
    except ValueError as erro:
        raise ValueError(f"Velocidade inválida: {velocidade_txt}") from erro

    return {
        "participante": participante,
        "condicao": condicao,
        "velocidade_kmh": velocidade,
    }


def codigo_dispositivo(nome_dispositivo: str) -> str:
    """Retorna apenas o código entre parênteses no nome do sensor."""
    resultado = re.search(r"\(([^)]+)\)", str(nome_dispositivo))
    return resultado.group(1) if resultado else str(nome_dispositivo)


def identificar_perna(nome_dispositivo: str) -> str:
    """Identifica a perna pelo final do código do sensor."""
    codigo = codigo_dispositivo(nome_dispositivo).upper()
    if codigo.endswith(SUFIXO_DIREITA.upper()):
        return "D"
    if codigo.endswith(SUFIXO_ESQUERDA.upper()):
        return "E"
    return "Desconhecida"


def filtrar_sinal(sinal: np.ndarray, corte_hz: float) -> np.ndarray:
    """Aplica Butterworth passa-baixa de quarta ordem com fase zero."""
    sos = butter(
        ORDEM_FILTRO,
        corte_hz,
        btype="lowpass",
        fs=FS,
        output="sos",
    )
    return sosfiltfilt(sos, np.asarray(sinal, dtype=float))


def filtrar_eixo_z(acc_z: np.ndarray) -> np.ndarray:
    """Suaviza o eixo anteroposterior somente para detectar os ciclos."""
    return filtrar_sinal(acc_z, CORTE_HZ)


def detectar_intervalos_passada(acc_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Detecta um máximo por ciclo no eixo Z filtrado.

    Retorna
    -------
    indices_picos : ndarray
        Índices das amostras correspondentes aos máximos.
    intervalos : ndarray
        Intervalos entre máximos consecutivos, em segundos.
    """
    sinal_filtrado = filtrar_eixo_z(acc_z)
    distancia_amostras = int(round(DISTANCIA_MIN_S * FS))

    indices_picos, _ = find_peaks(
        sinal_filtrado,
        distance=distancia_amostras,
        prominence=PROEMINENCIA_MIN_G,
    )
    intervalos = np.diff(indices_picos) / FS
    return indices_picos, intervalos


def selecionar_trecho_central(serie: np.ndarray, n: int = N_DFA) -> np.ndarray:
    """Seleciona n valores centrais, mantendo a ordem temporal."""
    serie = np.asarray(serie, dtype=float)
    if len(serie) < n:
        raise ValueError(
            f"A série possui {len(serie)} intervalos; são necessários {n}."
        )
    inicio = (len(serie) - n) // 2
    return serie[inicio : inicio + n]


def calcular_dfa_128(serie: np.ndarray) -> dict:
    """
    Calcula alpha-DFA para N=128.

    Parâmetros utilizados:
    - n_min = 4;
    - n_max = N/2 = 64;
    - escalas em progressão geométrica: 4, 8, 16, 32 e 64;
    - detrending polinomial de ordem 2.
    """
    x = np.asarray(serie, dtype=float)
    if len(x) != N_DFA:
        raise ValueError(f"A DFA exige exatamente {N_DFA} valores.")
    if not np.all(np.isfinite(x)):
        raise ValueError("A série contém valores ausentes ou infinitos.")

    perfil = np.cumsum(x - np.mean(x))
    flutuacoes = []

    for escala in ESCALAS_DFA:
        residuos_quadrados = []

        # Como 128 é divisível por todas as escalas, não há sobra de dados.
        for inicio in range(0, N_DFA, escala):
            segmento = perfil[inicio : inicio + escala]
            tempo_local = np.arange(escala, dtype=float)
            coeficientes = np.polyfit(
                tempo_local,
                segmento,
                deg=ORDEM_DETREND_DFA,
            )
            tendencia = np.polyval(coeficientes, tempo_local)
            residuos_quadrados.extend((segmento - tendencia) ** 2)

        f_n = np.sqrt(np.mean(residuos_quadrados))
        flutuacoes.append(f_n)

    flutuacoes = np.asarray(flutuacoes)
    if np.any(flutuacoes <= 0) or not np.all(np.isfinite(flutuacoes)):
        raise ValueError("Não foi possível calcular flutuações DFA válidas.")

    ajuste = linregress(np.log(ESCALAS_DFA), np.log(flutuacoes))
    return {
        "alpha_dfa": float(ajuste.slope),
        "dfa_r2": float(ajuste.rvalue**2),
        "escalas": ESCALAS_DFA.copy(),
        "flutuacoes": flutuacoes,
    }


def calcular_variabilidade_forma_onda(
    sinal_filtrado: np.ndarray,
    indices_picos: np.ndarray,
) -> dict:
    """
    Quantifica quanto uma forma de onda varia entre 100 passadas.

    O sinal deve ter sido previamente filtrado. Cada ciclo é interpolado para
    101 pontos. A medida é a média do desvio-padrão entre os ciclos ao longo
    de 0--100% da passada, normalizada pela amplitude pico a pico da curva
    média.
    """
    if len(indices_picos) < N_CICLOS_FORMA_ONDA + 1:
        raise ValueError(
            f"São necessários ao menos {N_CICLOS_FORMA_ONDA + 1} eventos "
            "para a forma de onda."
        )

    inicio = (len(indices_picos) - (N_CICLOS_FORMA_ONDA + 1)) // 2
    picos_centrais = indices_picos[
        inicio : inicio + N_CICLOS_FORMA_ONDA + 1
    ]

    sinal = np.asarray(sinal_filtrado, dtype=float)
    destino = np.linspace(0.0, 1.0, PONTOS_POR_CICLO)
    ciclos = []

    for pico_inicial, pico_final in zip(picos_centrais[:-1], picos_centrais[1:]):
        segmento = sinal[pico_inicial : pico_final + 1]
        if len(segmento) < 4:
            raise ValueError(
                "Foi encontrado um ciclo curto demais para interpolação."
            )
        origem = np.linspace(0.0, 1.0, len(segmento))
        ciclos.append(np.interp(destino, origem, segmento))

    matriz_ciclos = np.asarray(ciclos, dtype=float)
    curva_media = np.mean(matriz_ciclos, axis=0)
    dp_ponto_a_ponto = np.std(matriz_ciclos, axis=0, ddof=1)
    variabilidade_g = float(np.mean(dp_ponto_a_ponto))
    amplitude_curva_media = float(np.ptp(curva_media))

    if amplitude_curva_media <= 0:
        raise ValueError("A curva média não possui amplitude válida.")

    correlacoes = np.corrcoef(matriz_ciclos)
    triangulo_superior = correlacoes[np.triu_indices_from(correlacoes, k=1)]
    correlacao_media = float(np.nanmean(triangulo_superior))

    return {
        "variabilidade_g": variabilidade_g,
        "variabilidade_pct": 100.0 * variabilidade_g / amplitude_curva_media,
        "amplitude_curva_media_g": amplitude_curva_media,
        "correlacao_media_ciclos": correlacao_media,
    }


def nomear_metricas_forma_onda(resumo: dict, sufixo: str) -> dict:
    """Adiciona Y ou resultante aos nomes das métricas de forma de onda."""
    return {
        f"variabilidade_forma_onda_{sufixo}_g": resumo["variabilidade_g"],
        f"variabilidade_forma_onda_{sufixo}_pct": resumo["variabilidade_pct"],
        f"amplitude_curva_media_{sufixo}_g": resumo["amplitude_curva_media_g"],
        f"correlacao_media_ciclos_{sufixo}": resumo[
            "correlacao_media_ciclos"
        ],
    }


def processar_perna(
    dados_perna: pd.DataFrame,
    nome_arquivo: str,
    metadados: dict,
    nome_dispositivo: str,
) -> dict:
    """Processa um sensor/perna e devolve uma linha de resultados."""
    aceleracoes = {}
    for coluna in [
        COLUNA_MEDIO_LATERAL,
        COLUNA_LONGITUDINAL,
        COLUNA_ANTEROPOSTERIOR,
    ]:
        aceleracoes[coluna] = pd.to_numeric(
            dados_perna[coluna],
            errors="coerce",
        ).to_numpy()
        if np.any(~np.isfinite(aceleracoes[coluna])):
            raise ValueError(
                f"{coluna} possui valores ausentes ou não numéricos."
            )

    acc_x = aceleracoes[COLUNA_MEDIO_LATERAL]
    acc_y = aceleracoes[COLUNA_LONGITUDINAL]
    acc_z = aceleracoes[COLUNA_ANTEROPOSTERIOR]

    indices_picos, intervalos = detectar_intervalos_passada(acc_z)
    intervalos_plausiveis = (
        (intervalos >= INTERVALO_MIN_S)
        & (intervalos <= INTERVALO_MAX_S)
    )
    qc_intervalos = bool(np.all(intervalos_plausiveis))
    tem_100_ciclos = len(intervalos) >= N_CICLOS_FORMA_ONDA
    tem_n_dfa = len(intervalos) >= N_DFA

    resultado = {
        **metadados,
        "arquivo": nome_arquivo,
        "dispositivo": codigo_dispositivo(nome_dispositivo),
        "perna": identificar_perna(nome_dispositivo),
        "n_amostras": len(acc_z),
        "duracao_nominal_s": len(acc_z) / FS,
        "n_eventos": len(indices_picos),
        "n_intervalos_disponiveis": len(intervalos),
        "intervalos_plausiveis_pct": 100.0 * np.mean(intervalos_plausiveis)
        if len(intervalos)
        else 0.0,
        "qc_intervalos": qc_intervalos,
        "tem_100_ciclos_forma_onda": tem_100_ciclos,
        "tem_128_intervalos_dfa": tem_n_dfa,
        "acc_x_min_g": float(np.min(acc_x)),
        "acc_x_max_g": float(np.max(acc_x)),
        "acc_y_min_g": float(np.min(acc_y)),
        "acc_y_max_g": float(np.max(acc_y)),
        "acc_z_min_g": float(np.min(acc_z)),
        "acc_z_max_g": float(np.max(acc_z)),
        "tempo_passada_medio_s": np.nan,
        "dp_tempo_passada_s": np.nan,
        "cv_tempo_passada_pct": np.nan,
        "cadencia_passos_min": np.nan,
        "variabilidade_forma_onda_y_g": np.nan,
        "variabilidade_forma_onda_y_pct": np.nan,
        "amplitude_curva_media_y_g": np.nan,
        "correlacao_media_ciclos_y": np.nan,
        "variabilidade_forma_onda_resultante_g": np.nan,
        "variabilidade_forma_onda_resultante_pct": np.nan,
        "amplitude_curva_media_resultante_g": np.nan,
        "correlacao_media_ciclos_resultante": np.nan,
        "alpha_dfa": np.nan,
        "dfa_r2": np.nan,
    }

    # Não calcula medidas automaticamente quando há pico provavelmente perdido.
    # Remover ou interpolar intervalos alteraria a organização temporal da série.
    if not qc_intervalos or not tem_100_ciclos:
        return resultado

    serie_100 = selecionar_trecho_central(
        intervalos,
        n=N_CICLOS_FORMA_ONDA,
    )

    # A forma de onda principal usa a aceleração longitudinal da tíbia (Y).
    # Para a sensibilidade, os três eixos são filtrados da mesma forma antes
    # do cálculo amostra a amostra da magnitude resultante.
    acc_x_forma = filtrar_sinal(acc_x, CORTE_FORMA_ONDA_HZ)
    acc_y_forma = filtrar_sinal(acc_y, CORTE_FORMA_ONDA_HZ)
    acc_z_forma = filtrar_sinal(acc_z, CORTE_FORMA_ONDA_HZ)
    acc_resultante = np.sqrt(
        acc_x_forma**2 + acc_y_forma**2 + acc_z_forma**2
    )

    forma_onda_y = nomear_metricas_forma_onda(
        calcular_variabilidade_forma_onda(acc_y_forma, indices_picos),
        sufixo="y",
    )
    forma_onda_resultante = nomear_metricas_forma_onda(
        calcular_variabilidade_forma_onda(acc_resultante, indices_picos),
        sufixo="resultante",
    )

    media = float(np.mean(serie_100))
    dp = float(np.std(serie_100, ddof=1))
    resultado.update(
        {
            "tempo_passada_medio_s": media,
            "dp_tempo_passada_s": dp,
            "cv_tempo_passada_pct": 100.0 * dp / media,
            "cadencia_passos_min": 120.0 / media,
            **forma_onda_y,
            **forma_onda_resultante,
        }
    )

    # A DFA é exploratória e só é calculada quando existem 128 intervalos.
    if tem_n_dfa:
        serie_128 = selecionar_trecho_central(intervalos, n=N_DFA)
        dfa = calcular_dfa_128(serie_128)
        resultado.update(
            {
                "alpha_dfa": dfa["alpha_dfa"],
                "dfa_r2": dfa["dfa_r2"],
            }
        )
    return resultado


def processar_arquivo(caminho: Path) -> list[dict]:
    """Processa os dois sensores contidos em um arquivo."""
    metadados = interpretar_nome_arquivo(caminho)
    metadados["grupo"] = GRUPOS.get(metadados["participante"], "Não informado")

    dados = pd.read_csv(caminho, sep="\t", low_memory=False)
    colunas_obrigatorias = {
        "DeviceName",
        COLUNA_MEDIO_LATERAL,
        COLUNA_LONGITUDINAL,
        COLUNA_ANTEROPOSTERIOR,
    }
    ausentes = colunas_obrigatorias.difference(dados.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes: {sorted(ausentes)}")

    resultados = []
    for nome_dispositivo, dados_perna in dados.groupby("DeviceName", sort=False):
        resultados.append(
            processar_perna(
                dados_perna=dados_perna.reset_index(drop=True),
                nome_arquivo=caminho.name,
                metadados=metadados,
                nome_dispositivo=str(nome_dispositivo),
            )
        )
    return resultados


def criar_media_bilateral(resultados_perna: pd.DataFrame) -> pd.DataFrame:
    """Gera uma observação por participante e condição."""
    validos = resultados_perna.loc[
        resultados_perna["variabilidade_forma_onda_y_pct"].notna()
        & resultados_perna["perna"].isin(["D", "E"])
    ].copy()

    chaves = ["participante", "grupo", "condicao", "velocidade_kmh"]
    colunas_media = [
        "tempo_passada_medio_s",
        "dp_tempo_passada_s",
        "cv_tempo_passada_pct",
        "cadencia_passos_min",
        "variabilidade_forma_onda_y_g",
        "variabilidade_forma_onda_y_pct",
        "amplitude_curva_media_y_g",
        "correlacao_media_ciclos_y",
        "variabilidade_forma_onda_resultante_g",
        "variabilidade_forma_onda_resultante_pct",
        "amplitude_curva_media_resultante_g",
        "correlacao_media_ciclos_resultante",
        "alpha_dfa",
        "dfa_r2",
    ]

    medias = (
        validos.groupby(chaves, as_index=False, observed=True)[colunas_media]
        .mean()
    )
    contagem = (
        validos.groupby(chaves, as_index=False, observed=True)["perna"]
        .nunique()
        .rename(columns={"perna": "n_pernas"})
    )
    medias = medias.merge(contagem, on=chaves, how="left")
    medias["qc_bilateral"] = medias["n_pernas"].eq(2)

    contagem_dfa = (
        validos.groupby(chaves, as_index=False, observed=True)["alpha_dfa"]
        .count()
        .rename(columns={"alpha_dfa": "n_pernas_dfa"})
    )
    medias = medias.merge(contagem_dfa, on=chaves, how="left")
    medias["qc_dfa_bilateral"] = medias["n_pernas_dfa"].eq(2)
    medias.loc[~medias["qc_dfa_bilateral"], ["alpha_dfa", "dfa_r2"]] = np.nan

    medias["condicao"] = pd.Categorical(
        medias["condicao"],
        categories=ORDEM_CONDICOES,
        ordered=True,
    )
    return medias.sort_values(["participante", "condicao"]).reset_index(drop=True)


# =============================================================================
# GRÁFICOS E ESTATÍSTICA
# =============================================================================

def grafico_individual_e_medias(dados: pd.DataFrame, destino: Path) -> None:
    """Mostra forma de onda Y, resultante e DFA exploratória."""
    dados = dados.loc[dados["qc_bilateral"]].copy()
    if dados.empty:
        return

    x = np.arange(len(ORDEM_CONDICOES))
    variaveis = [
        (
            "variabilidade_forma_onda_y_pct",
            "Variabilidade longitudinal Y (%)",
        ),
        (
            "variabilidade_forma_onda_resultante_pct",
            "Variabilidade da resultante 3D (%)\n(sensibilidade)",
        ),
        ("alpha_dfa", "α-DFA (exploratória)"),
    ]
    fig, eixos = plt.subplots(1, 3, figsize=(16, 5))

    for eixo, (variavel, rotulo_y) in zip(eixos, variaveis):
        for grupo, dados_grupo in dados.groupby("grupo", observed=True):
            cor = CORES_GRUPOS.get(grupo, "#555555")

            for _, participante in dados_grupo.groupby("participante"):
                serie = (
                    participante.set_index("condicao")[variavel]
                    .reindex(ORDEM_CONDICOES)
                )
                eixo.plot(x, serie, color=cor, alpha=0.25, linewidth=1.2)

            medias = (
                dados_grupo.groupby("condicao", observed=True)[variavel]
                .mean()
                .reindex(ORDEM_CONDICOES)
            )
            eixo.plot(
                x,
                medias,
                marker="o",
                markersize=7,
                linewidth=2.8,
                color=cor,
                label=grupo,
            )

        eixo.set_xticks(x, ["−10%", "Preferida", "+10%"])
        eixo.set_xlabel("Condição de velocidade")
        eixo.set_ylabel(rotulo_y)
        eixo.grid(alpha=0.25)

    eixos[2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    eixos[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(destino, dpi=300, bbox_inches="tight")
    plt.close(fig)


def ajustar_modelo_misto(dados: pd.DataFrame, variavel: str) -> str:
    """
    Ajusta modelo com grupo, condição e interação como efeitos fixos,
    e participante como intercepto aleatório.
    """
    if not STATSMODELS_DISPONIVEL:
        return (
            "Modelo misto não ajustado porque statsmodels não está instalado.\n"
            "No Anaconda Prompt, instale com: conda install statsmodels\n"
            "Alternativa com pip: pip install statsmodels\n"
        )

    analise = dados.loc[
        dados["qc_bilateral"]
        & dados[variavel].notna()
        & dados["grupo"].isin(["Experiente", "Inexperiente"])
    ].copy()

    if analise["participante"].nunique() < 4:
        return (
            "Modelo não ajustado: inclua pelo menos quatro participantes e "
            "mais de um participante em cada grupo.\n"
        )
    if (analise.groupby("grupo")["participante"].nunique() < 2).any():
        return "Modelo não ajustado: é necessário mais de um participante por grupo.\n"

    formula = (
        f'{variavel} ~ C(grupo, Treatment(reference="Inexperiente")) * '
        'C(condicao, Treatment(reference="PREF"))'
    )

    try:
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            modelo = mixedlm(
                formula,
                data=analise,
                groups=analise["participante"],
                re_formula="1",
            )
            ajuste = modelo.fit(reml=False, method="lbfgs", maxiter=2000)

        texto = ajuste.summary().as_text()
        if avisos:
            texto += "\n\nAVISOS DO AJUSTE:\n"
            texto += "\n".join(f"- {aviso.message}" for aviso in avisos)
        return texto + "\n"
    except Exception as erro:
        return f"Não foi possível ajustar o modelo misto: {erro}\n"


def testes_pos_hoc(dados: pd.DataFrame, variavel: str) -> pd.DataFrame:
    """
    Comparações de uma variável com correção de Holm.

    - Grupos são comparados em cada condição com teste t de Welch.
    - Condições são comparadas de forma pareada dentro de cada grupo.
    """
    analise = dados.loc[
        dados["qc_bilateral"]
        & dados[variavel].notna()
        & dados["grupo"].isin(["Experiente", "Inexperiente"])
    ].copy()

    testes = []

    # Comparação entre grupos em cada velocidade.
    for condicao in ORDEM_CONDICOES:
        subconjunto = analise.loc[analise["condicao"] == condicao]
        experientes = subconjunto.loc[
            subconjunto["grupo"] == "Experiente", variavel
        ].to_numpy()
        inexperientes = subconjunto.loc[
            subconjunto["grupo"] == "Inexperiente", variavel
        ].to_numpy()
        if len(experientes) >= 2 and len(inexperientes) >= 2:
            teste = ttest_ind(experientes, inexperientes, equal_var=False)
            testes.append(
                {
                    "familia": "grupos_em_cada_condicao",
                    "comparacao": f"Experiente vs Inexperiente em {condicao}",
                    "estatistica": teste.statistic,
                    "p_bruto": teste.pvalue,
                }
            )

    # Comparações pareadas entre condições dentro de cada grupo.
    pares = [("-10", "PREF"), ("PREF", "+10"), ("-10", "+10")]
    for grupo in ["Experiente", "Inexperiente"]:
        amplo = (
            analise.loc[analise["grupo"] == grupo]
            .pivot(index="participante", columns="condicao", values=variavel)
        )
        for condicao_a, condicao_b in pares:
            if condicao_a not in amplo or condicao_b not in amplo:
                continue
            pares_validos = amplo[[condicao_a, condicao_b]].dropna()
            if len(pares_validos) >= 2:
                teste = ttest_rel(
                    pares_validos[condicao_a],
                    pares_validos[condicao_b],
                )
                testes.append(
                    {
                        "familia": f"condicoes_{grupo}",
                        "comparacao": f"{condicao_a} vs {condicao_b}",
                        "estatistica": teste.statistic,
                        "p_bruto": teste.pvalue,
                    }
                )

    resultado = pd.DataFrame(testes)
    if resultado.empty:
        return resultado

    resultado["p_holm"] = np.nan
    for _, indices in resultado.groupby("familia").groups.items():
        indices = list(indices)
        valores_p = resultado.loc[indices, "p_bruto"].to_numpy(dtype=float)
        ordem = np.argsort(valores_p)
        m = len(valores_p)
        ajustados_ordenados = (m - np.arange(m)) * valores_p[ordem]
        ajustados_ordenados = np.maximum.accumulate(ajustados_ordenados)
        ajustados_ordenados = np.minimum(ajustados_ordenados, 1.0)
        ajustados = np.empty(m, dtype=float)
        ajustados[ordem] = ajustados_ordenados
        resultado.loc[indices, "p_holm"] = ajustados
    return resultado


# =============================================================================
# EXECUÇÃO
# =============================================================================

def main() -> None:
    PASTA_RESULTADOS.mkdir(parents=True, exist_ok=True)
    arquivos = sorted(PASTA_DADOS.glob("*.txt"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo .txt encontrado em {PASTA_DADOS}")

    todas_as_pernas = []
    arquivos_ignorados = []

    for caminho in arquivos:
        try:
            resultados = processar_arquivo(caminho)
            todas_as_pernas.extend(resultados)
            print(f"OK: {caminho.name}")
        except Exception as erro:
            arquivos_ignorados.append((caminho.name, str(erro)))
            print(f"IGNORADO: {caminho.name} -> {erro}")

    if not todas_as_pernas:
        raise RuntimeError("Nenhum arquivo válido foi processado.")

    resultados_perna = pd.DataFrame(todas_as_pernas)
    resultados_perna.to_csv(
        PASTA_RESULTADOS / "resultados_por_perna.csv",
        index=False,
        encoding="utf-8-sig",
        decimal=",",
        sep=";",
    )

    medias_bilaterais = criar_media_bilateral(resultados_perna)
    medias_bilaterais.to_csv(
        PASTA_RESULTADOS / "resultados_media_bilateral.csv",
        index=False,
        encoding="utf-8-sig",
        decimal=",",
        sep=";",
    )

    grafico_individual_e_medias(
        medias_bilaterais,
        PASTA_RESULTADOS / "grafico_forma_onda_e_dfa.png",
    )

    texto_modelo_forma_onda = ajustar_modelo_misto(
        medias_bilaterais,
        "variabilidade_forma_onda_y_pct",
    )
    (PASTA_RESULTADOS / "modelo_misto_forma_onda.txt").write_text(
        texto_modelo_forma_onda,
        encoding="utf-8",
    )

    texto_modelo_resultante = ajustar_modelo_misto(
        medias_bilaterais,
        "variabilidade_forma_onda_resultante_pct",
    )
    (
        PASTA_RESULTADOS / "modelo_misto_resultante_sensibilidade.txt"
    ).write_text(
        texto_modelo_resultante,
        encoding="utf-8",
    )

    texto_modelo_alpha = ajustar_modelo_misto(medias_bilaterais, "alpha_dfa")
    (PASTA_RESULTADOS / "modelo_misto_alpha_exploratorio.txt").write_text(
        texto_modelo_alpha,
        encoding="utf-8",
    )

    pos_hoc_forma_onda = testes_pos_hoc(
        medias_bilaterais,
        "variabilidade_forma_onda_y_pct",
    )
    pos_hoc_forma_onda.to_csv(
        PASTA_RESULTADOS / "pos_hoc_forma_onda.csv",
        index=False,
        encoding="utf-8-sig",
        decimal=",",
        sep=";",
    )

    pos_hoc_resultante = testes_pos_hoc(
        medias_bilaterais,
        "variabilidade_forma_onda_resultante_pct",
    )
    pos_hoc_resultante.to_csv(
        PASTA_RESULTADOS / "pos_hoc_resultante_sensibilidade.csv",
        index=False,
        encoding="utf-8-sig",
        decimal=",",
        sep=";",
    )

    if arquivos_ignorados:
        pd.DataFrame(
            arquivos_ignorados,
            columns=["arquivo", "motivo"],
        ).to_csv(
            PASTA_RESULTADOS / "arquivos_ignorados.csv",
            index=False,
            encoding="utf-8-sig",
            sep=";",
        )

    print("\nProcessamento concluído.")
    print(f"Resultados salvos em: {PASTA_RESULTADOS}")
    print("\nMédias bilaterais:")
    print(
        medias_bilaterais[
            [
                "participante",
                "grupo",
                "condicao",
                "velocidade_kmh",
                "cadencia_passos_min",
                "cv_tempo_passada_pct",
                "variabilidade_forma_onda_y_g",
                "variabilidade_forma_onda_y_pct",
                "correlacao_media_ciclos_y",
                "variabilidade_forma_onda_resultante_g",
                "variabilidade_forma_onda_resultante_pct",
                "correlacao_media_ciclos_resultante",
                "alpha_dfa",
                "dfa_r2",
                "qc_bilateral",
                "qc_dfa_bilateral",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
