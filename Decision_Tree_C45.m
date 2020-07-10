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
root = best_node(selected_data)
% Laço que subdivide a árvore até chegar às folhas   
%node_children = raise_nodes(root)


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

% Função que divide o dataset presente no nó de acordo com a condição
% presente no nó
function [data1, data2] = split_data(node)
    
    data = node.data; % Dados do nó antes da divisão
    feature = node.feature; % Índice do atributo utilzizado para divisão
    bias = node.bias; % Threshold utilizado para divisão
    
    % Divisão dos dados
    data1 = data(data(:, feature) <= bias, :);
    data2 = data(data(:, feature) > bias, :);
end

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
            
            % Cálculo do ganho de informação
            gain = information_gain(dataset, data1, data2);
            
            node.data = dataset;
            
            if gain > best_gain
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
    root_node.type = 'r' % r para root
    root_node.depth = 0;
    
end

% Função que espande os nós
function node_children = raise_nodes(node_father)
    
    data_father = node_father.data;
    feature = node_father.feature;
    bias = node_father.bias;
    depth = node_father.depth;
    
    % Inicializando o nó filho 1
    node_1.type = 'm'; % m quer dizer nó intermediário
    node_1.depth = depth+1;
    
    % Inicializando o nó filho 1
    node_2.type = 'm'; % m quer dizer nó intermediário
    node_2.depth = depth+1;
    
    [node_1.data, node_2.data] = split_data(node_father);
    
    node_children = [node_1 node_2];
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
function gain = information_gain(dataset, split1, split2)
    initial_entropy = entropy(dataset);
    ent1 = entropy(split1);
    ent2 = entropy(split2);
    
    gain = initial_entropy - (ent1+ent2)/2;
end