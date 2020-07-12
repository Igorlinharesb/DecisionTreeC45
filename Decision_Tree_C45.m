% Universidade Federal do Cear� - UFC
% Reconhecimento de  Padr�es - 2020.1
% Francisco Igor Fel�cio Linhares - 374874

% Decision Tree C4.5 implementado e avaliados com Cross-Validation

% Comando para ver tempo de execu��o do script
tic;

% Inicializando o ambiente de trabalho
clc;
clear;
close all;

% Carregando a base de dados e movendo a coluna target da primeira pra
% �ltima coluna
data = readmatrix('data/wine.csv');
data = [data(:, 2:end) data(:, 1)];

% Nome dos atributos da base de dados
features = {'�lcool', '�cido m�lico', 'Cinza', 'Alcalinidade da cinza',...
            'Magn�sio', 'Fen�is Totais', 'Flavon�ides', ...
            'Fen�is n�o flavon�ides', 'Proantocianidinas', ...
            'Intensidade da cor', 'Matiz', 'OD280/OD315', 'Prolina'};

% Plotando histograma dos dados para an�lise
histograms(data, features)

% A partir dos histogramas, escolhi os atributos �lcool, Cinza, Flavon�ides,
% Intensidade de cor e Prolina, por aparentarem ser bons discriminantes
selected_data = data(:, [1 3 7 10 13 14]);
selected_names = features([1 3 7 10 13]);

% Plotando gr�ficos de dispers�o
pairplot(selected_data, selected_names)

% Testa �rvore com os atributos selecionados � priori
result = cross_validation(selected_data, 10);

% Faz teste com �rvore criada com atributos que n�o aparentam ser bons
% discriminantes
result2 = cross_validation(data(:, [2 4 5 8 14]), 10);

% Faz teste com �rvore criada com todos os atributos
result3 = cross_validation(data, 10);

% Mostra as precis�es m�dias na janela de comando
avg_result = mean(result)
avg_result2 = mean(result2)
avs_result3 = mean(result3)

toc;
% --------------------------- FUN��ES -------------------------------------

% Fun��o que faz a plotagem em pares, recebe como par�metro a matriz de
% atributos e os nomes dos atributos.
function histograms(dataset, feature_names)
    
    n = length(feature_names);
    classes = unique(dataset(:,end)); % Vari�veis target
   
    % Quantidade de linhas de gr�ficos
    rows = ceil(n/3);
    
    figure('Name', 'Histogramas');
    for i=1:n
        bins = linspace(min(dataset(:, i)), max(dataset(:, i)), 25);
        counts = zeros(length(bins), length(classes));
        
        for j=1:length(classes)
           class_data = dataset(dataset(:, end)==classes(j), i);
           counts(:, j) = histc(class_data, bins);
        end
        
        subplot(rows,3, i);
        bar(bins, counts, 'stacked');
        title(feature_names(i));
        hold off;
    end
end


% Fun��o que realiza a plotagem dos scatter plots dos atributos
% selecionados
function pairplot(dataset, feature_names)

    m = length(dataset(1,1:end-1)); % N�mero de atributos
    
    classes = unique(dataset(:, end));
    
    figure('Name', 'Gr�ficos de Dispers�o');
    % La�o que povoa os subplots
    for i=2:m % Linhas
        for j=1:i-1 % Colunas
            subplot(m-1, m-1, (m-1)*(i-2) + j)
            for k=1:length(classes)
                class_data = dataset(dataset(:, end)==k, :);
                plot(class_data(:, j), class_data(:, i), '.',...
                     'MarkerSize', 9);
                hold on;
            end
            
            % Adicionando label nos eixos horizontais da �ltima linha
            if i==m
                xlabel(feature_names(j))
            end
            
            % Adicionando label nos eixos verticais da primeira coluna
            if j==1
                ylabel(feature_names(i))
            end
        end
        hold off;
    end
end


% Fun��o que encontra a condi��o que melhor separa os dados e retorna o n�
function node = best_node(dataset)
    
    best_gain = 0;
    
    for feature=1:size(dataset, 2)-1
        % Para cada atributo ser�o utilizados 100 valores igualmente
        % espa�ados para avaliar o ganho de infoma��o para cada valor
        range = max(dataset(:, feature)) - min(dataset(:, feature));
        divisions = 100;
        pace = range/divisions;
        
        % Testando 100 valores do atributo para separar o dataset
        for i=1:divisions-1
            bias = min(dataset(:, feature))+ i*pace;
            
            % Divis�o dos dados
            data1 = dataset(dataset(:, feature) <= bias, :);
            data2 = dataset(dataset(:, feature) > bias, :);
            
            % C�lculo da entropia inicial e do ganho de informa��o
            [initial_entropy, gain] = information_gain(dataset, data1, data2);
            
            % Cria o n� com as melhores caracter�sticas
            node.data = dataset;
            if gain > best_gain
                node.entropy = initial_entropy;
                node.feature = feature;
                node.bias = bias;
                node.gain = gain;
                best_gain = gain;
            end
        end
    end
end


% Fun��o que encontra o n� raiz da �rvore
function root_node = find_root(dataset)
    
    root_node = best_node(dataset);
    root_node.type = 'r'; % r para root
    root_node.depth = 0;
    root_node.expanded = 0;
    
end


% Fun��o que retorna zero se os dados representam um n� intermedi�rio e
% retorna a classe no caso de ser uma folha
function isleaf = is_leaf(data)
    % Para identificar se o n� constitui uma folha utilizei 2 crit�rios:
    % 1 - se alguma classe representa mais de 80% dos dados daquele n�
    % 2 - se o n� possui 4 amostras ou menos, � retornada a classe com mais
    % amostras nos dados do n�.
    
    % Inicializa a vari�vel de retorno
    isleaf = 0;
    
    % Calcula as probabilidades das classes nos dados do n�
    classes = unique(data(:, end));
    prior_probs = zeros(length(classes), 1);
    
    for i=1: length(classes)
        % Calcula a probabilidade a prior de cada classe
        count = length(data(data(:,end) == classes(i)));
        prior_probs(i) = count/size(data, 1);
        
        % Verifica se alguma das classes aparece em mais de 80% dos dados e
        % retorna o valor da classe predominante
        if prior_probs(i) > 0.8
            isleaf = classes(i);
        end
        
        % Verifica se o n� possui poucos dados e retorna a classe com maior
        % probabilidade
        if size(data, 1) < 5 && prior_probs(i) == max(prior_probs)
            isleaf = classes(i);
        end
    end
end


% Fun��o que expande a �rvore
function node_father = expand_tree(node_father)
    
    % Extraindo os dados do n� pai
    data_father = node_father.data; % Dados do n�
    feature = node_father.feature; % Atributo utilizado na divis�o
    bias = node_father.bias; % bias
    depth = node_father.depth; % Profundidade do n�
    
    % Divis�o dos dados e atribui��o aos n�s filhos
    data_1 = data_father(data_father(:, feature) <= bias, :);
    data_2 = data_father(data_father(:, feature) > bias, :);
    
    % Iniciando com o n� 1
    % Verifica se os dados constituem uma folha ou um n� intermedi�rio
    isleaf = is_leaf(data_1);
    
    if isleaf == 0
        % Caso o n�o seja uma folha � encontrado o melhor crit�rio de
        % divis�o
        node_1 = best_node(data_1);
        node_1.type = 'm';  % Altera o tipo do n� para middle (m)
        node_1.depth = depth+1;
        node_1.expanded = 1;
    else
        node_1.data = data_1;
        node_1.entopy = entropy(data_1);
        node_1.type = 'l'; % Altera o tipo do n� para leaf (l)
        node_1.depth = depth+1;
        node_1.class = isleaf;
    end
    
    % Repetindo o processo para o n� 2
    isleaf = is_leaf(data_2);
    
    if isleaf == 0
        node_2 = best_node(data_2);
        node_2.type = 'm';
        node_2.depth = depth+1;
        node_2.expanded = 0;
    else
        node_2.data = data_2;
        node_2.entopy = entropy(data_2);
        node_2.type = 'l'; % l para folha
        node_2.depth = depth+1;
        node_2.class = isleaf;
    end
    
    % Avaliando o n� 1
    if node_1.type == 'l'
        % Se o n� for uma folha, a divis�o � interrompida
        node_father.child_1 = node_1;
    else
        % Se n�o for folha continua expandindo o n�
        node_father.child_1 = expand_tree(node_1);
    end
    
    % Avaliando o n� 2
    if node_2.type == 'l'
        node_father.child_2 = node_2;
    else
        node_father.child_2 = expand_tree(node_2);
    end
    
    node_father.expanded = 1;
end


% Fun��o que calcula a entropia
function result = entropy(dataset)
    
    % Classes da base de dados
    classes = unique(dataset(:, end));
    
    % Calcular a entropia
    result = 0;
    for i=1: length(classes)
        % N�mero de amostras da classe
        count = length(dataset(dataset(:,end) == classes(i)));
        prior_prob = count/length(dataset); % probabilidade � priori
        result = result - prior_prob*log2(prior_prob); % Entropia
    end
end


% Fun��o que calcula o ganho de informa��o
function [initial_entropy, gain] = information_gain(dataset, split1, split2)
    
    % Entropia inicial
    initial_entropy = entropy(dataset);
    
    % Entropia das divis�es
    ent1 = entropy(split1);
    ent2 = entropy(split2);
    
    % Ganho de informa��o
    gain = initial_entropy - (ent1+ent2)/2;
end


% Fun��o que efetua a classifica��o de uma amostra:
function prediction = predict(tree, sample)

    % Verificando se a �rvore se resume � uma folha
    if tree.type == 'l'
        prediction = tree.class;
    else
        % Verificando para qual n� filho a amostra vai ser direcionada
        if sample(tree.feature) <= tree.bias
            prediction = predict(tree.child_1, sample);
        else
            prediction = predict(tree.child_2, sample);
        end
    end
end


% Fun��o que recebe dados de treino e teste, cria �rvore de decis�o com
% base no dataset de treino, faz a previs�o das amostras de teste e retorna
% a precis�o das previs�es.
function score = evaluate_tree(train, test)
    
    % Cria �rvore com base nos dados de treino
    root = find_root(train);
    tree = expand_tree(root);
    
    % Realiza as previs�es para cada amostra de teste e conta os acertos
    hits = 0;
    for i=1: length(test)
        prediction = predict(tree, test(i, 1:end-1));
        if prediction == test(i, end)
            hits = hits+1;
        end
    end
    score = hits/length(test);     
end


% Cross validation que retorna um array com a precis�o em cada divis�o
% treino-teste.
function scores = cross_validation(dataset, folds)
    
    % Inicializando o vetor de precis�es
    scores = zeros(1, folds);
    
    % Retorna o tamanho das parti��es de teste
    partitions = cvpartition(length(dataset), 'Kfold', 10);
    test_sizes = partitions.TestSize;
    
    % Inicializando o vetor de �ndices aleat�rios
    indexes = randperm(length(dataset), length(dataset));
    
    b = 0;
    e = 0;
    
    for i=1:folds
        
        % As vari�veis b e e ajudar�o a controlar em que posi��o os �ndices 
        % de teste iniciam e terminam em indexes
        
       % Dividindo os data sets de treino e teste
       temp_indexes = indexes;
       
       % Separando base para teste
       e = e + test_sizes(i);
       test_indexes = temp_indexes(b+1:e);
       test = dataset(test_indexes, :);
       
       % Atribuindo o restante para treino
       temp_indexes(b+1:e) = [];
       train = dataset(temp_indexes, :);
       b = e+1;       
       
       % Avalia��o do modelo
       scores(i) = evaluate_tree(train, test);
    end
end