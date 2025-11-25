# KWS (Keyword Spotting) com MFCC na Raspberry Pi Zero 2W

Este projeto implementa um sistema de Keyword Spotting (KWS) otimizado para a Raspberry Pi Zero 2W. Foram treinados diferentes tipos de modelos para classificação de áudio com o Google Speech Dataset.

Esses repositório está separado em pastas que contém diferentes etapas e modelos de todo o projeto. O nome das pastas indica qual etapa foi feita ou qual tipo de modelo foi treinado. As pastas de modelo listadas abaixo contém cada uma os arquivos dos modelos treinados, um jupyter notebook com o passo a passo do treinamento, os arquivos com os requirements das bibliotecas do Python para aquela aplicação e o script para inferência.

## Requisitos de Hardware

* Raspberry Pi Zero 2W
* Cartão Micro SD
* Microfone (USB ou I2S compatível com a Pi)
* (Opcional) LED ou outro atuador para testes de GPIO

## Requisitos de Software

* Python 3.11
* Jupyter Notebook / Jupyter Lab
* Raspberry Pi OS
* Outros requisitos podem ser baixados usando os comandos abaixo nos respectivos venvs:

```bash
# No PC para realizar o treinamento:
pip install -r requirements-pc.txt

# Na placa de desenvolvimento para realizar a inferência:
pip install -r requirements-rasp.txt
```

Observação: A instalação do tflite-runtime na Pi Zero 2W pode falhar a depender da versão do numpy. Recomenda-se baixar exatamente as versões usadas no projeto para não haver erros. Além disso, existem pacotes externos que precisam ser baixados caso ainda não tenham sido, esses são:

* portaudio19-dev
* python3-dev
