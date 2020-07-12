# DecisionTreeC45

## Objetivo
Implementação de algoritmo de Árvore de decisão C4.5, utilizando Ganho de Informação como função de custo, para classificação da base de dados de vinho UCI Wine Dataset utilizando pelo menos 5 atributos no MATLAB e validação com Cross Validation.

## Base de dados
A base de dados é constituída de 178 amostras de vinho divididas em 3 classes. Cada amostra possui 13 atributos.

## Escolha dos atributos
Para selecionar os atributos, plotei os histogramas de cada atributo para cada classe para analisar a separabilidade das classes, conforme a imagem baixo.

![Histogramas](https://github.com/Igorlinharesb/DecisionTreeC45/blob/master/images/histograms.png)

 A partir da observação dos histogramas acima selecionei os atributos Álcool, Cinza, Flavonóides, Intensidade de cor e Prolina por aparentarem ser bons discriminantes para a base de dados.
 
 Para entender melhor comos os atributos se relacionam e comos as classes podem ser separadas plotei os gráficos de dispersão abaixo:
 ![Pairplots](https://github.com/Igorlinharesb/DecisionTreeC45/blob/master/images/scatter_plots.png)
 
 ## Implementação
 Cada nó da árvore é representado como uma *struct* do MATLAB, e podem ser de 3 tipos: raiz, intermediário e folha e cada tipo tem os atributos listados abaixo.
 
 * **Type:** O tipo do nó: *r* se for um nó raiz, *m* se for um nó intermediário e *l* se for um nó folha.
 * **Data:** Os dados de treino presentes naquele nó.
 * **Entropy:** Entropia dos dados presentes naquele nó.
 * **Feature:** O índice do atributo que melhor divide as classes.
 * **Bias:** O valor da feature que melhor divide as classes.
 * **Depth:** A profundidade do nó.
 * **Class:** Se o nó for uma folha esse atributo armazena a classe do nó.
 
 No caso de nós raizes e intermediários eles também possuem os atributos

 * **Gain:** O ganho de informação daquele nó após a divisão dos dados.
 * **Expanded:** Variável de controle para saber se o nó foi expandido ou não, e só aparece em nós raiz e nós intermediários.
 * **Child_1:** Nó filho que recebe a primeira divisão dos dados.
 * **Child_2:** Nó filho que recebe a segunda divisão dos dados.
 
 E os nós folhas possuem um atributo **Class** que armazena a classe do nó. 
 
 A implementação do algoritmo se resume em Treino, Teste e Validação.
 
 ### Treino
 1. Encontrar o melhor atributo e melhor bias e criar o nó raiz, de forma que maximize o ganho de informação
 2. O nó raiz é expandido gerando 2 novos nós.
 3. Os novos nós são verificados e caso não constituam nós folha são expandidos, e assim sucessivamente.
 4. Os nós param de ser expandidos quando pelo menos 80% dos dados do nó pertence à uma classe, ou se no nó tem menos de 5 amostras, nesse caso o nó recebe a classe predominante.
 
 ### Teste
 Após ter a árvore treinada, algumas amostras de teste são utilizadas para validar o modelo.
 1. A amostra é avaliada pelo nó raiz, e se o atributo da amostra for menor que o atributo do nó raiz ela é classificada para o nó *Child_1*, caso contrário é classificada para o nó *Child_2*.
 2. O processo se repete até que a amostra chegue em um nó folha.
 3. A amostra é classificada com a classe do nó folha.
 
 ## Validação e Resultados
 O modelo foi avaliado com *Cross-Validation* com 10 *folds*, em que para cada divisão treino-teste é criada uma árvore com o dados de treino e avaliada com os dados de teste e calculada a precisão das previsões. 
 
 Para validar o modelo, fiz testes utilizando diferentes atributos para a classificação. 
 * No primeiro caso utilizei os atributos que havia inicialmente selecionado como bons discriminantes, e o modelo atingiu uma precisão média de **92.94%**;
 * No segundo caso utilizei todos os atributos fornecidos para a classificação e foi obtida uma precisão média de **89.89%**;
 * No último caso utilizei os atributos que não aparentavam ser bons discriminantes, sendo eles: Ácido málico, Alcalinidade da cinza, Magnésio e Fenóis não flavonóides, e obtendo uma precisão média de **68.57%**.
 
 O modelo levou cerca de 20 segundos para executar.
