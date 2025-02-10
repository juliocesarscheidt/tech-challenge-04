# IA-para-Devs

Código com implementação do Tech Challenge fase 4

O código utiliza as bibliotecas mediapipe e deepface, para detecção de poses e emoções respectivamente. São implementadas algumas lógicas para detecção das emoções de forma contínua e detecção de atividades através dos landmarks fornecidos pela mediapipe. No final é gerado um sumário com tais índices.

### Execução

Para analisar um vídeo, adicione-o na pasta do projeto com o nome "video.mp4". Então baixe as dependências com o pip:

```bash
pip install -r requirements.txt
```

Então execute o script main.py

```bash
python main.py
```

Com isto será gerado um vídeo de output com o nome "output_video.mp4" e um arquivo "summary.txt".
