# Sistemas de coordenada do Mundo, da Câmera e Projeção em perspectiva

Este projeto apresenta o desenvolvimento e aplicação de técnicas de modelagem e
transformações de sólidos em um sistema de coordenadas tridimensional utilizando a
linguagem Python. O objetivo principal é criar representações em arame de diversos sólidos
geométricos, como cilindro, cone, esfera, tronco de pirâmide e cubo, e posteriormente
posicionar esses sólidos em um cenário coerente, evitando sobreposições e intersecções
entre eles.

Na primeira parte, foram utilizadas funções para criar os sólidos a partir
de parâmetros fornecidos. Cada sólido foi modelado de forma indireta, levando em
consideração medidas como altura, raios das bases, número de fatias e circunferências
intermediárias. Foram criadas funções que retornam a matriz de vértices e arestas de cada
sólido, permitindo sua posterior manipulação no sistema de coordenadas tridimensional.

## Cilindro
![image](https://github.com/MateusMaccos/SistemaDeCoordenadasEProjecoes/assets/75508372/09b4da89-ed92-4c61-b61b-494f2fb5e5d3)
## Cone
![image](https://github.com/MateusMaccos/SistemaDeCoordenadasEProjecoes/assets/75508372/3454257a-bcc0-4add-b21a-85f2e716c784)
## Cubo
![image](https://github.com/MateusMaccos/SistemaDeCoordenadasEProjecoes/assets/75508372/47d15e85-4d7e-4992-bb14-04e942aeb138)
## Esfera
![image](https://github.com/MateusMaccos/SistemaDeCoordenadasEProjecoes/assets/75508372/7c83d3d1-f2be-4150-a217-906925f529da)
## Tronco de pirâmide
![image](https://github.com/MateusMaccos/SistemaDeCoordenadasEProjecoes/assets/75508372/fb51fa1b-a68f-4798-b720-681b50343252)

# Sistema de coordenadas do mundo
Em seguida, uma cena foi composta, onde os sólidos modelados anteriormente
foram posicionados no sistema de coordenadas do mundo. Foram aplicadas
transformações de escala, rotação e translação para evitar sobreposições e intersecções
entre os sólidos. O cilindro e o cone foram posicionados em um octante específico,
enquanto a esfera, o tronco de pirâmide e o cubo foram posicionados em outro octante.
Foi estabelecido um limite máximo para as componentes dos vértices no sistema de
coordenadas do mundo, definindo-o como 10. Quando necessário, foram aplicadas
transformações adicionais para garantir que os sólidos estivessem dentro desses limites.

![image](https://github.com/MateusMaccos/SistemaDeCoordenadasEProjecoes/assets/75508372/cee3562d-f2b3-41d1-a7e4-84aad1105356)

# Sistema de coordenadas da câmera
Em seguida, foi selecionado um octante vazio e um ponto de origem para o sistema
de coordenadas da câmera. A base vetorial (n, u e v) do novo sistema de coordenadas foi
calculada considerando o volume de visão, usando o ponto médio entre os centros de
massa de cada um dos sólidos. Os objetos no sistema de coordenadas do mundo foram
então transformados para o sistema de coordenadas da câmera.

![image](https://github.com/MateusMaccos/SistemaDeCoordenadasEProjecoes/assets/75508372/d2c29405-e1a4-4745-ac94-86814b0991d3)

A origem do sistema de coordenadas do mundo foi mostrada como um ponto no
sistema de coordenadas da câmera. Os diversos sólidos também foram apresentados
nesse sistema de coordenadas tridimensionais.

# Projeção em Perspectiva
Por fim, foi realizada uma transformação de projeção em perspectiva dos sólidos
contidos no volume de visão. As arestas dos sólidos foram projetadas em duas dimensões
na janela de projeção. Cada sólido recebeu uma cor única para suas arestas, permitindo
uma melhor visualização e distinção entre os objetos.

![image](https://github.com/MateusMaccos/SistemaDeCoordenadasEProjecoes/assets/75508372/a4be3345-8646-4707-9f89-2aff36469b30)

