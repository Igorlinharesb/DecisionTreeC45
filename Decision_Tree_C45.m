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

% Dividi o atributo Prolina por 1000 apenas para facilitar o acompanhamento
% pela janela de comando
selected_data(:, 5) = selected_data(:, 5)/1000;

% pair plots

% Criando a árvore
% root = find_root(selected_data) % Nó raiz
root = find_root(selected_data);


[root.child_1, root.child_2] = expand_node(root);
% Laço que subdivide a árvore até chegar às folhas   


% Cross validation
    
    
    
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
% function pairplots


% Função que encontra a melhor divisão
function node = best_node(dataset)
    
    best_gain = 0;
    
    for feature=1:size(dataset, 2)-1
        range = max(dataset(:, feature)) - min(dataset(:, feature));
        divisions = 10;
        pace = range/10;
        
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
    
end

% Função que retorna zero se os dados representam um nó 
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

% Função que espande os nós
function [node_1, node_2] = expand_node(node_father)
    
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
    else
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
    else
        node_2.entopy = entropy(data_2);
        node_2.type = 'l'; % l para folha
        node_2.depth = depth+1;
        node_2.class = isleaf;
    end
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