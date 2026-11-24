# Industrial Pose Analytics

Aplicação desktop para análise ergonômica industrial em vídeos, com detecção corporal offline por MediaPipe Tasks Pose Landmarker. A interface usa CustomTkinter e segue o painel escuro de cards do modelo visual fornecido.

## Recursos

- Dashboard com métricas da sessão, gráficos e estado do modelo; telas visitadas ficam em cache para uma navegação mais fluida.
- Seleção de vídeo MP4, AVI ou MOV, prévia e leitura de resolução, FPS, duração e frames.
- MediaPipe Pose Landmarker Full em modo `VIDEO`, com um operador por padrão.
- Esqueleto, pontos e rótulos de articulações principais desenhados com OpenCV.
- Processamento frame a frame em thread separada, progresso e cancelamento seguro.
- Exportação de vídeo MP4 anotado, CSV de landmarks e CSV de indicadores ergonômicos por frame.
- Indicadores geométricos: inclinação do tronco, elevação de cada braço, flexão de cada cotovelo, inclinação cabeça/tronco e flexão de cada joelho.
- Resumos por indicador com média, mediana, desvio padrão, mínimo, máximo, frames válidos/inválidos e cobertura calculada sobre os frames usados.
- Gráfico temporal comparando dados originais com filtros independentes de média móvel ou Savitzky-Golay; os CSVs permanecem sem suavização.
- Exportação de relatório PDF com identificação do vídeo, dados do processamento, estatísticas, gráficos temporais e limitações da análise.
- Análise retroativa do último CSV de landmarks existente para gerar os indicadores sem executar novamente o detector.
- Configuração de limiares de detecção, presença, tracking e visibilidade; logs locais em `logs/`.

Os ângulos são medidos em graus a partir da projeção 2D dos landmarks. A orientação da câmera influencia os valores; a aplicação apresenta medidas geométricas e não atribui classificações de risco ergonômico.
Os lados esquerdo e direito são anatômicos conforme os landmarks MediaPipe e podem parecer invertidos em vídeos frontais ou espelhados. As estatísticas requerem ao menos duas amostras válidas; valores ausentes ou não finitos não entram nos cálculos.

## Requisitos e ambiente

- Windows 10/11 e Python 3.11.
- Dependências listadas em `requirements.txt`.
- Modelo Pose Landmarker Full em `models/pose_landmarker_full.task`.

Ative o ambiente:

```powershell
.\.venv\Scripts\Activate.ps1
```

Para criar o ambiente em outra máquina:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Executar

```powershell
python main.py
```

Vídeos, landmarks e indicadores são gravados em `output/` por padrão. A tela Video Analysis permite escolher outra pasta.

## Validar

```powershell
python -m unittest discover -s tests -v
python -m pip check
```

Os testes cobrem a geometria dos indicadores, filtro de visibilidade e um processamento curto com o modelo MediaPipe real, incluindo exportação dos três arquivos.
