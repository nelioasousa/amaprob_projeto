# Aprendizagem de Máquina Probabilística: tópicos principais
1. Modelos/Soluções Bayesianas
2. Regressão Linear Bayesiana
    - Propriedades da distribuição Gaussiana: conjunta, marginal, condicional e lineariedade
    - Inferência sequencial (atualização Bayesiana) e em batch
    - Comparação Bayesiana de modelos: a evidência (verossimilhança marginal)
    - Máxima Verossimilhança II
3. Regressão Logística Bayesiana
    - Otimização Iterative Reweighted Least Squares (IRLS)
    - Aproximação de Laplace
    - Bayesian Information Criterion (BIC)
4. Modelos de Misturas
    - Variáveis latentes discretas
    - Mistura de Gaussianas (Gaussian Mixture Model)
    - Algoritmo Expectation-Maximization
5. Probabilistic Principal Component Analysis (PPCA)
    - Variáveis latentes contínuas
    - Geração de dados usando o PPCA
6. Inferência Variacional
    - Evidence Lower Bound (ELBO)
7. Processos Gaussianos
8. Variational Autoencoders
    - Deep Latent Variable Model (DLVM)
    - Truque da reparametrização (reparameterization trick)
9. **[bônus]** Diffusion Models
10. **[bônus]** Normalizing flows
11. **[bônus]** Flow matching
12. **[bônus]** Log-Gaussian Cox processes

# Projeto Final
## Problema 1
Estimação de densidade de probabilidade ou redução de dimensionalidade ou dados faltantes ou geração de dados.

**Contextualização:** temos um grande conjunto de imagens de pavimentos asfálticos de rodovias. A grande maioria das imagens não possuem defeitos (remendos ou buracos) e queremos explorar essas imagens para obter exemplos de defeitos. É muito fácil encontrar exemplos sem defeitos, porém os exemplos com defeitos são raros e a exploração manual é de difícil execução. Para auxiliar a exploração, podemos modelar a distribuição das imagens sem defeito e buscar de forma automática por anomalias segundo essa distribuição.

**Dados:** conjunto de imagens não rotuladas de pavimentos asfálticos de rodovias.

**Objetivo:** obter um subconjunto de imagens com defeitos para exploração e rotulação.

**Ideia:** usar uma CNN pré-treinada para extração de atributos e aplicar VAE e PPCA em cima dos atributos extraídos

### Metodologia

**Dataset:** O dataset real possui milhões de imagens de pavimentos asfálticos, porém nem todas estão disponíveis para download. Das imagens com acesso, foram selecionadas 6744 imagens em períodos do dia com sol, porém algumas tem incidência de sombras. Todas as 6744 imagens já são rotuladas para detecção de buracos, remendos e trincas. Das 6744 imagens, 6110 são sem defeitos e 634 com defeito. Cerca de 30% das imagens irão compor um conjunto de validação (os rótulos de com/sem defeito serão mantidos). Os 70% de imagens restantes serão utilizadas para simular um conjunto de dados completo e totalmente desconhecido (simular o conjunto com milhões de imagens e sem rótulo algum).

**Objetivo:** explorar um conjunto grande de imagens de pavimentos asfálticos de forma eficiente, buscando obter um subconjunto de imagens que tenham boa representatividade de exemplos com e sem defeitos.

**Contextualização:** não é possível (tratável) inspecionar visualmente milhões de imagens. Portanto, é necessário uma método inteligente para uma exploração inicial desses dados. Essa exploração inicial servirá para tomadas de decisões. Ademais, o subconjunto selecionado durante a exploração poderá servir como base para futuras explorações e/ou pré-treino de modelos discriminantes. Outras técnicas de exploração serão necessárias após essa etapa inicial, como, por exemplo, o uso de active learning. Além do mais, uma boa e representativa exploração inicial permitira melhores resultados e menores custos.

**Método:**

> a. Processamento dos dados

Uma vez que modelos específicos para imagens (CNNs, VAEs com encoder e decoder com camadas convolucionais, etc.) são custosos computacionalmente e um de nossos objetivos é ser eficiênte (computacional e temporalmente), não iremos trabalhar no domínio das imagens. Todas as imagens serão passadas por uma CNN pré-treinada (YOLOv11 pré-treinado no dataset de benchmark COCO) em um dataset genérico, extraindo vetores de características (bordas, texturas, cores, etc.) de cada uma. A hipótese é que esses vetores de características genérico consigam minimamente fornecer informações para discriminar imagens de pavimentos sem defeitos das imagens de pavimentos com defeitos.

As imagens possuem dimensão 1280x360. O modelo usado, YOLOv11n, extrai vetores de características de dimensão 256. Porém, cada imagem foi dividida em duas imagens de 640x360 antes da extração dos vetores, de modo que cada imagem foi substituida por um vetor de características de dimensão 512.

> b. Separação do conjunto de validação

As 6744 imagens advêm de vários segmentos de vídeos diversos. Portanto, há imagens que são sequênciais (por exemplo, os frames 50, 51, ..., 60 de um mesmo vídeo) e consequentemente possuem características visuais muito semelhantes. A separação do conjunto de validação levou isso em consideração, de modo que imagens em sequência sempre fiquem juntas em um mesmo grupo. O conjunto de validação é composto por 30% da sequências que não apresentam defeito e 30% das sequências que apresentam defeito.

> c. Exploração das imagens

O passo-a-passo abaixo detalha o procedimento executado:

1) Selecionar aleatoriamente 50 imagens do conjunto;
2) Contabilizar quantas imagens possuem defeitos;
3) Adicionar as imagens limpas ao conjunto de imagens limpas descobertas;
4) Adicionar as imagens com defeitos ao conjunto de imagens com defeito descobertas;
5) Estimar a distribuição das imagens limpas usando o conjunto de imagens limpas descobertas;
6) Avaliar a qualidade da distribuição modelada usando o conjunto de validação;
7) Usar a distribuição modelada para selecionar 50 novas imagens, selecionando aquelas com menor probabilidade;
8) Voltar ao passo **2.** e repetir até obter 1000 imagens no total.

> d. O que não será feito

- Não haverá otimização de hiperparâmetros (tamanho da dimensão latente da projeção do PPCA/VAE, camadas do VAE, passo de aprendizagem do treinamento, etc.)
- Fine-tuning do modelo YOLO usado na extração de vetores de características
- Validação/avaliação do treinamento do VAE em cada iteração (referente ao passo **5.**)

## Problema 2
Regressão com predições probabilística.

**Contextualização:** um sistema de aluguel de bicicletas precisa alocar (estações e quantidade de bibicletas por estação) de forma inteligente a sua frota de acordo com as demandas de seus usuários. São várias as variáveis que influenciam a demanda por bicicletas de aluguel, como variáveis socioeconômicas, clima e tempo, horário, dia, entre outros.

**Dataset:** [Bike sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

**Objetivo:** estimar a quantidade de bicicletas alugadas em um certo dia.

**Modelos:** Regressão Linear Bayesiana e Gaussian Process.
