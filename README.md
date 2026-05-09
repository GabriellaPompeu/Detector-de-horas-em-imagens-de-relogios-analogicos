# Detector de horas em imagens de relógios analógicos
O objetivo do projeto é realizar a análise de imagens de relógios e determinar o horário indicado.

## Pipeline de processamento do projeto:
Primeiramente, é feita a detecção de círculos na imagem para que se possa identificar o relógio e redimensioná-lo. Assim, podemos aplicar uma função de identificação de contornos na nova imagem e identificar as linhas que estão mais perto do centro, que serão os candidatos a ponteiros. Portanto, extraindo os ângulos dos ponteiros, obteremos a hora correspondente a imagem de entrada.

## Repositório
O projeto é baseado em YANG, Charig; XIE, Weidi; ZISSERMAN, Andrew. It’s about time: Analog clock reading in the wild. 2021 (Disponível em: <http://arxiv.org/abs/2111.09162>) e, por isso, foi utilizado o mesmo repositório.
