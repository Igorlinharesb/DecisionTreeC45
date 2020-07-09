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

features = {'Álcool', 'Ácido málico', 'Cinza', 'Alcalinidade da cinza',...
            'Magnésio', 'Fenóis Totais', 'Flavonoides', ...
            'Fenóis não flavonóides', 'Proantocianidinas', ...
            'Intensidade da cor', 'Matiz', 'OD280/OD315', 'Prolina'};
        
histograms(data, features)

% --------------------------- FUNÇÕES -------------------------------------
% Função que faz a plotagem em pares, recebe como parâmetro a matriz de
% atributos e os nomes dos atributos.
function histograms(dataset, feature_names)

    m = length(feature_names) % Número de atributos
    classes = unique(dataset(:,end));
    colors = [0 0.4470 0.7410;0.8500 0.3250 0.0980;0.9290 0.6940 0.1250];
    rows = ceil(m/3);
    
    figure('Name', 'Histogramas');
    % Laço que povoa os subplots
    for i=1:m
        subplot(rows,3, i);
        for j=1:length(classes)
            class_data = dataset(dataset(:, end)==classes(j), :);
            histogram(class_data(:,i), 10, 'FaceColor', colors(j,:), ...
                        'FaceAlpha', 0.7);
            hold on;
        end
        title(feature_names(i));
        hold off;
    end
end