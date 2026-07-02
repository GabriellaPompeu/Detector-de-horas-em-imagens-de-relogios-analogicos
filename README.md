# Detector de horas em imagens de relógios analógicos :mantelpiece_clock:
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

## Pipeline de processamento do projeto: :watch:
O pipeline desenvolvido é dividido em oito etapas principais:
- Detecção e segmentação do relógio;
- Remoção do fundo;
- Pré-processamento;
- Detecção do mostrador circular;
- Detecção dos ponteiros;
- Identificação dos ponteiros corretos;
- Conversão angular para tempo.

## Desafios encontrados: :stopwatch:
Estes foram os desafios encontrados e solucionados com sucesso parcial.
- localizar o relógio na imagem
- remover ruídos do fundo
- detectar o círculo do mostrador
- identificar ponteiros
- separar hora/minuto/segundos
- converter orientação angular em tempo


## Tutorial :alarm_clock:
Para utilizar este código em seu computador, siga estes passos:
 1. Clone o repositório para o seu computador.
 2. Obtenha imagens de relógios analógicos. Para os nossos testes, utilizamos algumas imagens do dataset fornecido na pasta "data" do repositório [itsabouttime]( https://github.com/charigyang/itsabouttime/tree/main), utilizado como referência para este trabalho.
 3. Salve as imagens em uma pasta denominada "imagens". Esta pasta deve estar salva no mesmo local que o arquivo .py do código foi salvo em seu computador.
 4. Utilizando a IDE de sua preferência, abra o código e rode o arquivo "projeto.py".
 5. Digite a letra "a" e pressione "Enter" para visualisar as leituras dos relógios individualmente.
 6. Primeiro, aparece o desenho de um quadrado azul na imagem original indicando a detecção do relógio pelo YOLO.
 7. Para prosseguir, feche esta página.
 8. Automaticamente, uma nova página se abrirá com a leitura do horário do relógio.
 9. O ponteiro das horas é indicado pela linha azul e o ponteiro dos minutos, pela linha vermelha. Ambas aparecem dentro do círculo verde, que indica onde a leitura está sendo feita.
 10. Aparecem no terminal, durante a leitura do relógio, alguns dados como: detecção de relógio quadrado, detecção insuficiente de ponteiros, detecção do número "12", etc.
 11. Feche a página novamente para ir para a próxima imagem de relógio.
 12. Continue fechando as páginas até finalizar todas as suas imagens.


## Referência :timer_clock:
O projeto é baseado em YANG, Charig; XIE, Weidi; ZISSERMAN, Andrew. It’s about time: Analog clock reading in the wild. 2021 (Disponível em: <http://arxiv.org/abs/2111.09162> e https://github.com/charigyang/itsabouttime/tree/main) e, por isso, foi utilizado o mesmo repositório.
