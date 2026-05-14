# Detector de horas em imagens de relógios analógicos
O objetivo do projeto é implementar uma solução híbrida para leitura automática de relógios analógicos. A arquitetura usa YOLO para segmentação e Visão Computacional Clássica para interpretação geométrica.
O modelo YOLO realiza:
- detecção do relógio;
- segmentação do objeto.

A visão computacional clássica realiza:
- remoção do fundo;
- detecção do círculo;
- extração dos ponteiros;
- identificação de hora e minuto;
- conversão angular para horário.

## Pipeline de processamento do projeto:
O pipeline desenvolvido é dividido em oito etapas principais:
- Detecção e segmentação do relógio;
- Remoção do fundo;
- Pré-processamento;
- Detecção do mostrador circular;
- Detecção dos ponteiros;
- Identificação dos ponteiros corretos;
- Conversão angular para tempo.

## Desafios encontrados:
Estes foram os desafios encontrados e solucionados com sucesso parcial.
- localizar o relógio na imagem
- remover ruídos do fundo
- detectar o círculo do mostrador
- identificar ponteiros
- separar hora/minuto/segundos
- converter orientação angular em tempo

## Repositório
O projeto é baseado em YANG, Charig; XIE, Weidi; ZISSERMAN, Andrew. It’s about time: Analog clock reading in the wild. 2021 (Disponível em: <http://arxiv.org/abs/2111.09162> e https://github.com/charigyang/itsabouttime/tree/main) e, por isso, foi utilizado o mesmo repositório.
