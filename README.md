# **A Rota do Neuromante - Classificação de armas de fogo pelo disparo** 

Este projeto tem como objetivo a implementação e comparação de algoritmos de aprendizado de máquina para a classificação de tipos de armas através de sinais sonoros. Utilizando técnicas de processamento de áudio, foram extraídas características como MFCCs e centróide espectral para alimentar modelos preditivos.

Os modelos implementados foram:

  - **Baseline**: Dummy Classifier
  - **Random Forest** (Floresta Aleatória)
  - **MLP** (Multi Layer Perceptron) utilizando PyTorch Lightning 

## **Para utilizar o código**

1.  **Pré-requisitos**:
    Certifique-se de ter o Python instalado e as seguintes bibliotecas:

    ```bash
    pip install numpy pandas matplotlib scikit-learn librosa optuna torch lightning torchmetrics
    ```

2.  **Dataset**:

      - O código espera que os arquivos de áudio estejam na pasta `./guns_dataset`.
      - Baixe o dataset [Gunshot Audio Dataset no Kaggle](https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset/data) e extraia-o no diretório do projeto.
      - *Nota:* Conforme observado no notebook, a pasta `M16` foi removida para evitar redundância com o modelo `M4`.

3.  **Execução**:

      - Abra o arquivo `Neuromante.ipynb` em um ambiente Jupyter.
      - Execute as células em ordem para realizar a extração de características, otimização de hiperparâmetros com o **Optuna** e o treinamento dos modelos.

## **Introdução**

A partir de características sonoras, como centro de massa, frequências e amplitude, este trabalho visa prever o tipo de arma utilizada em um disparo. A métrica principal de otimização é o **f1-macro**, escolhida por sua adequação a problemas multiclasse, garantindo peso igual para todas as categorias independentemente do número de amostras. Técnicas de processamento de sinais como contraste espectral e coeficientes cepstrais em escala Mel (MFCC) foram fundamentais para a representação dos dados.

## **Fonte dos dados**

O dataset utilizado foi:

  - AYDEMIR, E. **Gunshot Audio Dataset**. Kaggle, 2021. Disponível em: [https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset/data](https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset/data).

## **Conclusão**

Após o processo de otimização de hiperparâmetros, os modelos apresentaram os seguintes desempenhos nos dado de teste (lembrando que o processo de treino do MLP é estocástico e os resultado podem ter uma leve divergência):

  - **MLP**: Melhor estimativa de f1-macro de aproximadamente **0.9040**.
  - **Random Forest**: Melhor estimativa de f1-macro de aproximadamente **0.9460**.

## **Referências**

O trabalho fundamentou-se nas seguintes bibliotecas e materiais didáticos:

[1] AYDEMR, Emrah. **Gunshot Audio Dataset**. [S. l.]: Kaggle, 2023. Disponível em: [https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset/data](https://www.kaggle.com/datasets/emrahaydemr/gunshot-audio-dataset/data). Acesso em: 5 abr. 2026.

[2] **MEL-FREQUENCY cepstrum**. In: WIKIPEDIA: the free encyclopedia. [S. l.]: Wikimedia Foundation, 2024. Disponível em: [https://en.wikipedia.org/wiki/Mel-frequency\_cepstrum](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum). Acesso em: 5 abr. 2026.

[3] **MEL scale**. In: WIKIPEDIA: the free encyclopedia. [S. l.]: Wikimedia Foundation, 2024. Disponível em: [https://en.wikipedia.org/wiki/Mel\_scale](https://en.wikipedia.org/wiki/Mel_scale). Acesso em: 5 abr. 2026.

[4] **SPECTRAL centroid**. In: WIKIPEDIA: the free encyclopedia. [S. l.]: Wikimedia Foundation, 2024. Disponível em: [https://en.wikipedia.org/wiki/Spectral\_centroid](https://en.wikipedia.org/wiki/Spectral_centroid). Acesso em: 5 abr. 2026.

[5] LIBROSA. **librosa.feature.spectral\_contrast**. [S. l.], 2024. Disponível em: [https://librosa.org/doc/main/generated/librosa.feature.spectral\_contrast.html](https://librosa.org/doc/main/generated/librosa.feature.spectral_contrast.html). Acesso em: 5 abr. 2026.

[6] **ZERO-crossing rate**. In: WIKIPEDIA: the free encyclopedia. [S. l.]: Wikimedia Foundation, 2024. Disponível em: [https://en.wikipedia.org/wiki/Zero-crossing\_rate](https://en.wikipedia.org/wiki/Zero-crossing_rate). Acesso em: 5 abr. 2026.

[7] **CHROMA feature**. In: WIKIPEDIA: the free encyclopedia. [S. l.]: Wikimedia Foundation, 2024. Disponível em: [https://en.wikipedia.org/wiki/Chroma\_feature](https://en.wikipedia.org/wiki/Chroma_feature). Acesso em: 5 abr. 2026.

[8] **MULTILAYER perceptron**. In: WIKIPEDIA: the free encyclopedia. [S. l.]: Wikimedia Foundation, 2024. Disponível em: [https://en.wikipedia.org/wiki/Multilayer\_perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron). Acesso em: 5 abr. 2026.

[9] PYTORCH. **CrossEntropyLoss**. [S. l.], 2024. Disponível em: [https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Acesso em: 5 abr. 2026.

[10] PRADHAN, R. **The Math Behind Adam Optimizer**. Towards Data Science, 2018. Disponível em: [https://towardsdatascience.com/the-math-behind-adam-optimizer-c41407efe59b/](https://towardsdatascience.com/the-math-behind-adam-optimizer-c41407efe59b/). Acesso em: 5 abr. 2026.

[11] HUGHES, Chris P. **A brief overview of Cross Entropy Loss**. Medium, 2023. Disponível em: [https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5](https://medium.com/@chris.p.hughes10/a-brief-overview-of-cross-entropy-loss-523aa56b75d5). Acesso em: 5 abr. 2026.

[12] PYTORCH. **CrossEntropyLoss**. [S. l.], 2024. Disponível em: [https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html). Acesso em: 5 abr. 2026.

[13] CASSAR, Daniel Roberto. **Multilayer Perceptron em Python puro**. [Jupyter Notebook], Ilum – Escola de Ciência, Campinas, 2026.

[14] CASSAR, Daniel Roberto. **Construindo e treinando redes neurais com PyTorch e Lightning**. [Jupyter Notebook], Ilum – Escola de Ciência, Campinas, 2026.

[15] **PRINCIPAL component analysis**. In: WIKIPEDIA: the free encyclopedia. [S. l.]: Wikimedia Foundation, 2024. Disponível em: [https://en.wikipedia.org/wiki/Principal\_component\_analysis](https://en.wikipedia.org/wiki/Principal_component_analysis). Acesso em: 5 abr. 2026.

[16] TUNCER, T., DOGAN, S., AKBAL, E., AYDEMIR, E. (2021). **An Automated Gunshot Audio Classification Method Based On Finger Pattern Feature Generator And Iterative Relieff Feature Selector**, Journal of Engineering Science of Adıyaman University.

-----

**Autor**: Arthur Brandão do Nascimento
