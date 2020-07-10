% Universidade Federal do Cear� - UFC
% Reconhecimento de  Padr�es - 2020.1
% Francisco Igor Fel�cio Linhares - 374874

% Decision Tree C4.5implementado e avaliados com Cross-Validation

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
% histograms(data, features)

% A partir dos histogramas, escolhi os atributos �lcool, Cinza, Flavon�ides,
% Intensidade de cor e Prolina, por aparentarem ser bons discriminantes
selected_data = data(:, [1 3 7 10 13 14]);

% Dividi o atributo Prolina por 1000 apenas para facilitar o acompanhamento
% pela janela de comando
selected_data(:, 5) = selected_data(:, 5)/1000;

% pair plots

% Criando a �rvore
% root = find_root(selected_data) % N� raiz
root = best_node(selected_data)
% La�o que subdivide a �rvore at� chegar �s folhas   
%node_children = raise_nodes(root)


% Cross validation
    
    
    
% --------------------------- FUN��ES -------------------------------------

% Fun��o que faz a plotagem em pares, recebe como par�metro a matriz de
% atributos e os nomes dos atributos.
function histograms(dataset, feature_names)

    m = length(feature_names); % N�mero de atributos
    classes = unique(dataset(:,end)); % Vari�veis target
    % Cores para plotagem
    colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980;0.9290 0.6940 0.1250];
    % Quantidade de linhas de plotagens
    rows = ceil(m/3);
    
    figure('Name', 'Histogramas');
    % La�o que povoa os subplots
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

% Fun��o que realiza a plotagem dos scatter plots dos atributos
% selecionados
% function pairplots

% Fun��o que divide o dataset presente no n� de acordo com a condi��o
% presente no n�
function [data1, data2] = split_data(node)
    
    data = node.data; % Dados do n� antes da divis�o
    feature = node.feature; % �ndice do atributo utilzizado para divis�o
    bias = node.bias; % Threshold utilizado para divis�o
    
    % Divis�o dos dados
    data1 = data(data(:, feature) <= bias, :);
    data2 = data(data(:, feature) > bias, :);
end

% Fun��o que encontra a melhor divis�o
function node = best_node(dataset)
    
    best_gain = 0;
    
    for feature=1:size(dataset, 2)-1
        range = max(dataset(:, feature)) - min(dataset(:, feature));
        divisions = 10;
        pace = range/10;
        
        % Testando 10 valores do atributo para separar o dataset
        for i=1:divisions-1
            bias = min(dataset(:, feature))+ i*pace;
            
            % Divis�o dos dados
            data1 = dataset(dataset(:, feature) <= bias, :);
            data2 = dataset(dataset(:, feature) > bias, :);
            
            % C�lculo do ganho de informa��o
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

% Fun��o que encontra o n� raiz da �rvore
function root_node = find_root(dataset)
    
    root_node = best_node(dataset);
    root_node.type = 'r' % r para root
    root_node.depth = 0;
    
end

% Fun��o que espande os n�s
function node_children = raise_nodes(node_father)
    
    data_father = node_father.data;
    feature = node_father.feature;
    bias = node_father.bias;
    depth = node_father.depth;
    
    % Inicializando o n� filho 1
    node_1.type = 'm'; % m quer dizer n� intermedi�rio
    node_1.depth = depth+1;
    
    % Inicializando o n� filho 1
    node_2.type = 'm'; % m quer dizer n� intermedi�rio
    node_2.depth = depth+1;
    
    [node_1.data, node_2.data] = split_data(node_father);
    
    node_children = [node_1 node_2];
end

% Fun��o que calcula a entropia
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

% Fun��o que calcula o ganho de informa��o
function gain = information_gain(dataset, split1, split2)
    initial_entropy = entropy(dataset);
    ent1 = entropy(split1);
    ent2 = entropy(split2);
    
    gain = initial_entropy - (ent1+ent2)/2;
end