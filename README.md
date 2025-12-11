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
## Subprojeto 1
Estimação de densidade de probabilidade ou redução de dimensionalidade ou dados faltantes ou geração de dados.

### Ideia 1
**Dataset:** [HSV data](https://archive.ics.uci.edu/dataset/571/hcv+data)

**Tarefas:**
- inputação de dados faltantes
- estimação de densidade
- geração de dados

### Ideia 2

**Dataset:** imagens de pavimentos asfálticos com rótulos de detecção de buracos e remendos

**Tarefas:**
- embedding (extração de vetores de características) usando CNNs
- redução de dimensionalidade no espaço das embeddings
- agrupamento no espaço das embeddings (dimensão original e reduzida)

## Subprojeto 2
Regressão com predições probabilística.

### Ideia 1

**Dataset:** [Bike sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)

Predizer a quantidade de usuários casuais e registrados de um sistema de empréstimo de bicicletas. Útil para controlar a frota de bicletas disponíveis em uma localidade para um certo período.

### Ideia 2

**Dataset:** [Concrete Compressive Strength](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)

Predizer a força de compressão para uma certa mistura de concreto. Útil para auxiliar na confecção de misturas de concreto para obtenção de resistências desejadas.
