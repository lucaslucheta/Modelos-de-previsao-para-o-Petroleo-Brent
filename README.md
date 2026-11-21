================================================================================
BRENT QUANT TERMINAL — MODELAGEM ECONOMÉTRICA E PREVISÃO INSTITUCIONAL (BZ=F)
================================================================================

Aplicação interativa desenvolvida em Python e Streamlit para análise quantitativa,
previsão econométrica, backtesting out-of-sample e simulação de hedge corporativo
para o petróleo bruto ICE Brent (BZ=F).

--------------------------------------------------------------------------------
1. PRINCIPAIS FUNCIONALIDADES
--------------------------------------------------------------------------------
* Ingestão de Dados & Multi-Moeda Real:
  - Coleta automática das séries históricas diárias do Brent (BZ=F) via Yahoo Finance.
  - Conversão histórica exata (pregão a pregão) para 9 moedas: USD, BRL, EUR, GBP,
    JPY, CNY, CHF, CAD e AUD.
  - Integração de variáveis exógenas macro e físicas: Índice Dólar (DXY), WTI (CL=F),
    Crack Spread 3-2-1 (RB=F e HO=F) e Sazonalidade Cíclica (Fourier).

* Diagnóstico Estatístico de Regime (Z-Score):
  - Cálculo do preço de equilíbrio estrutural e bandas estacionárias (±2 desvios-padrão)
    com teste de estacionariedade Dickey-Fuller Aumentado (ADF).

* Modelo 1 — Projeção Tática de Curto e Médio Prazo (2 Anos / 504 dias úteis):
  - Ensemble regularizado (Ridge + BayesianRidge) combinado com atração de reversão
    à média estrutural.
  - Simulador de choques táticos de oferta (Disciplina da OPEP+ e Risco Geopolítico).
  - Cones de probabilidade assimétricos calibrados via Split Conformal Prediction:
    bandas de 50% (P25–P75), 80% (P10–P90) e 95% (P2.5–P97.5).

* Modelo 2 — Projeção Estrutural de Longo Prazo (Até 10 Anos):
  - Processo estocástico de reversão à média de Ornstein-Uhlenbeck (Schwartz-Smith)
    com saltos de Poisson (Jump-Diffusion) em 5.000 simulações de Monte Carlo.
  - Parâmetros ajustáveis para Custo Marginal Global de Produção, deriva secular de
    transição energética e intensidade de choques extremos.

* Validação Walk-Forward & Auditoria Multi-Janela:
  - Backtesting interativo sem vazamento de dados futuros (Out-of-Sample) com métricas
    de RMSE, MAPE, acurácia direcional e cobertura empírica das bandas.
  - Auditoria automática em 8 janelas históricas cobrindo ciclos de alta, queda e
    lateralização.

* Risco Financeiro (VaR / CVaR) & Simulador de Hedge:
  - Cálculo de Value-at-Risk (95% e 99%) e Expected Shortfall (CVaR) para produtores
    e consumidores.
  - Simulador de proteção de fluxo de caixa com comparação de distribuição antes e
    depois do hedge.

* Exportação de Dados:
  - Download direto em CSV das séries históricas ajustadas pelo câmbio e das
    projeções dos dois modelos.

--------------------------------------------------------------------------------
2. ESTRUTURA DO REPOSITÓRIO
--------------------------------------------------------------------------------
├── app.py              # Código principal da aplicação Streamlit e modelos quantitativos
├── requirements.txt    # Dependências Python necessárias para execução
└── readme.txt          # Documentação do projeto

--------------------------------------------------------------------------------
3. COMO INSTALAR E EXECUTAR LOCALMENTE
--------------------------------------------------------------------------------
Pré-requisitos: Python 3.9 ou superior instalado.

Passo 1 — Clone o repositório:
    git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
    cd SEU-REPOSITORIO

Passo 2 — (Opcional, recomendado) Crie e ative um ambiente virtual:
    python -m venv venv
    # No Windows:
    venv\Scripts\activate
    # No Linux/macOS:
    source venv/bin/activate

Passo 3 — Instale as dependências:
    pip install -r requirements.txt

Passo 4 — Inicie o terminal quantitativo:
    streamlit run app.py

O aplicativo será aberto automaticamente no navegador em http://localhost:8501.

---

NOTAS DE DESENVOLVIMENTO E DE APOIO TECNOLÓGICO
Este projeto foi concebido, estruturado e validado por Lucas Reis Lucheta. Durante o processo de desenvolvimento e programação da interface em Streamlit, foram utilizadas ferramentas de Inteligência Artificial (como o ChatGPT) como suporte técnico de mentoria, estruturação de código e boas práticas de desenvolvimento.

👨‍💻 AUTOR
Desenvolvido por Lucas Reis Lucheta
Estudante de Engenharia Química | EEL-USP
LinkedIn | GitHub
