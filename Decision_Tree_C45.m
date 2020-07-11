% Universidade Federal do Ceará - UFC
% Reconhecimento de  Padrões - 2020.1
% Francisco Igor Felício Linhares - 374874

% Decision Tree C4.5implementado e avaliados com Cross-Validation

% Comando para ver tempo de execução do script
tic;

% Inicializando o ambiente de trabalho
clc;
clear;
close all;

% Carregando a base de dados e movendo a coluna target da primeira pra
% última coluna
data = readmatrix('data/wine.csv');
data = [data(:, 2:end) data(:, 1)];

% Nome dos atributos da base de dados
features = {'Álcool', 'Ácido málico', 'Cinza', 'Alcalinidade da cinza',...
            'Magnésio', 'Fenóis Totais', 'Flavonóides', ...
            'Fenóis não flavonóides', 'Proantocianidinas', ...
            'Intensidade da cor', 'Matiz', 'OD280/OD315', 'Prolina'};

% Plotando histograma dos dados para análise
% histograms(data, features)

% A partir dos histogramas, escolhi os atributos Álcool, Cinza, Flavonóides,
% Intensidade de cor e Prolina, por aparentarem ser bons discriminantes
selected_data = data(:, [1 3 7 10 13 14]);

result = cross_validation(selected_data, 10);
avg_result = mean(result)

toc;
% --------------------------- FUNÇÕES -------------------------------------

% Função que faz a plotagem em pares, recebe como parâmetro a matriz de
% atributos e os nomes dos atributos.
function histograms(dataset, feature_names)

    m = length(feature_names); % Número de atributos
    classes = unique(dataset(:,end)); % Variáveis target
    % Cores para plotagem
    colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980;0.9290 0.6940 0.1250];
    % Quantidade de linhas de plotagens
    rows = ceil(m/3);
    
    figure('Name', 'Histogramas');
    % Laço que povoa os subplots
    for i=1:m
        subplot(rows,3, i);
        bins = linspace(min(dataset(:, i)), max(dataset(:, i)), 25);
        for j=1:length(classes)
            % Filtra os dados por classe
            class_data = dataset(dataset(:, end)==classes(j), :);
            % Plota o histograma de cada classe
            histogram(class_data(:,i), bins, 'FaceColor', colors(j,:), ...
                        'FaceAlpha', 0.7);
            hold on;
        end
        title(feature_names(i));
        hold off;
    end
end

% Função que realiza a plotagem dos scatter plots dos atributos
% selecionados
% function pairplots(dataset, feature_names)

% Função de visualização da árvore

% Função que encontra a condição que melhor separa os dados e retorna o nó
function node = best_node(dataset)
    
    best_gain = 0;
    
    for feature=1:size(dataset, 2)-1
        range = max(dataset(:, feature)) - min(dataset(:, feature));
        divisions = 100;
        pace = range/divisions;
        
        % Testando 10 valores do atributo para separar o dataset
        for i=1:divisions-1
            bias = min(dataset(:, feature))+ i*pace;
            
            % Divisão dos dados
            data1 = dataset(dataset(:, feature) <= bias, :);
            data2 = dataset(dataset(:, feature) > bias, :);
            
            % Cálculo da entropia inicial e do ganho de informação
            [initial_entropy, gain] = information_gain(dataset, data1, data2);
            
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

% Função que encontra o nó raiz da árvore
function root_node = find_root(dataset)
    
    root_node = best_node(dataset);
    root_node.type = 'r'; % r para root
    root_node.depth = 0;
    root_node.expanded = 0;
    
end

% Função que retorna zero se os dados representam um nó intermediário e
% retorna a classe no caso de ser uma folha
function isleaf = is_leaf(data)
    
    isleaf = 0;
    
    % Calcula as probabilidades das classes nos dados do nó
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
        
        % Verifica se o nó possui poucos dados e retorna a classe com maior
        % probabilidade
        if size(data, 1) < 5 && prior_probs(i) == max(prior_probs)
            isleaf = classes(i);
        end
    end
end

% Função que espande a árvore
function node_father = expand_tree(node_father)
    
    % Extraindo os dados do nó pai
    data_father = node_father.data;
    
    feature = node_father.feature;
    bias = node_father.bias;
    depth = node_father.depth;
    
    % Divisão dos dados e atribuição aos nós filhos
    data_1 = data_father(data_father(:, feature) <= bias, :);
    data_2 = data_father(data_father(:, feature) > bias, :);
    
    % Iniciando com o nó 1
    
    % Verifica se os dados constituem uma folha ou um nó intermediário
    isleaf = is_leaf(data_1);
    
    if isleaf == 0
        % Caso o não seja uma folha é encontrado o melhor critério de
        % divisão
        node_1 = best_node(data_1);
        node_1.type = 'm';
        node_1.depth = depth+1;
        node_1.expanded = 1;
    else
        node_1.data = data_1;
        node_1.entopy = entropy(data_1);
        node_1.type = 'l'; % l para folha
        node_1.depth = depth+1;
        node_1.class = isleaf;
    end
    
    % Repetindo o processo para o nó 2
    isleaf = is_leaf(data_2);
    
    if isleaf == 0
        % Caso o não seja uma folha é encontrado o melhor critério de
        % divisão
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
    
    if node_1.type == 'l'
        node_father.child_1 = node_1;
    else
        node_father.child_1 = expand_tree(node_1);
    end
    
    if node_2.type == 'l'
        node_father.child_2 = node_2;
    else
        node_father.child_2 = expand_tree(node_2);
    end
    
    node_father.expanded = 1;
end

% Função que calcula a entropia
function result = entropy(dataset)
    
    classes = unique(dataset(:, end));
    
    result = 0;
    % Calcular a entropia
    for i=1: length(classes)
        count = length(dataset(dataset(:,end) == classes(i)));
        prior_prob = count/length(dataset);
        result = result - prior_prob*log2(prior_prob);
    end
end

% Função que calcula o ganho de informação
function [initial_entropy, gain] = information_gain(dataset, split1, split2)
    initial_entropy = entropy(dataset);
    ent1 = entropy(split1);
    ent2 = entropy(split2);
    
    gain = initial_entropy - (ent1+ent2)/2;
end

% Função que efetua a classificação de uma amostra:
function prediction = predict(tree, sample)

    % Verificando se a árvore se resume à uma folha
    if tree.type == 'l'
        prediction = tree.class;
    else
        % Verificando para qual nó filho a amostra vai ser direcionada
        if sample(tree.feature) <= tree.bias
            prediction = predict(tree.child_1, sample);
        else
            prediction = predict(tree.child_2, sample);
        end
    end
end

% Função que recebe dados de treino e teste, cria árvore de decisão com
% base no dataset de treino, faz a previsão das amostras de teste e retorna
% a precisão das previsões.
function score = evaluate_tree (train, test)
    
    % Cria árvore com base nos dados de treino
    root = find_root(train);
    tree = expand_tree(root);
    
    % Realiza as previsões para cada amostra de teste e conta os acertos
    hits = 0;
    for i=1: length(test)
        prediction = predict(tree, test(i, 1:end-1));
        if prediction == test(i, end)
            hits = hits+1;
        end
    end
    score = hits/length(test);     
end

% Cross validation que retorna um array com a precisão em cada divisão
function scores = cross_validation(dataset, folds)
    
    % Inicializando o vetor de scores
    scores = zeros(1, folds);
    
    % Retorna o tamanho das partições de teste
    partitions = cvpartition(length(dataset), 'Kfold', 10);
    test_sizes = partitions.TestSize;
    
    % Inicializando o vetor de índices aleatórios
    indexes = randperm(length(dataset), length(dataset));
    
    % As variáveis b e e ajudarão a controlar em que posição os índices de 
    % teste iniciam e terminam em indexes
    b = 0;
    e = 0;
    
    for i=1:folds
        
       % Dividindo os data sets de treino e teste
       temp_indexes = indexes;
       
       % Teste
       e = e + test_sizes(i);
       test_indexes = temp_indexes(b+1:e);
       test = dataset(test_indexes, :);
       
       % Treino
       temp_indexes(b+1:e) = [];
       train = dataset(temp_indexes, :);
       b = e+1;       
       
       % Avaliação do modelo
       scores(i) = evaluate_tree(train, test);
    end
end