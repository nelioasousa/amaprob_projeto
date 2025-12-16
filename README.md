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

**Contextualização:** temos um grande conjunto de imagens de pavimentos asfálticos de rodovias. A grande maioria das imagens não possuem defeitos (remendos, buracos, trincas, outros) e queremos explorar essas imagens para obter exemplos de defeitos. É muito fácil encontrar exemplos sem defeitos, porém os exemplos com defeitos são raros e a exploração manual é de difícil execução. Para auxiliar a exploração, podemos modelar a distribuição das imagens sem defeito e buscar de forma automática por anomalias segundo essa distribuição.

**Dados:** conjunto de imagens não rotuladas de pavimentos asfálticos de rodovias.

**Objetivo:** obter um subconjunto de imagens com defeitos para exploração e rotulação.

**Ideias:**

1. Usar um VAE específico para imagens
2. Usar uma CNN pré-treinada para extração de atributos e aplicar VAE e PPCA em cima dos atributos extraídos

## Problema 2
Regressão com predições probabilística.

**Contextualização:** um sistema de aluguel de bicicletas precisa alocar (estações e quantidade de bibicletas por estação) de forma inteligênte a sua frota de acordo com as demandas de seus usuários. São várias as variáveis que influenciam a demanda por bicicletas de aluguel, como variáveis socioeconômicas, clima e tempo, horário, dia, entre outros.

**Dataset:** [Bike sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

**Objetivo:** estimar a quantidade de bicicletas alugadas em um certo dia.

**Modelos:** Regressão Linear Bayesiana e Gaussian Process.
