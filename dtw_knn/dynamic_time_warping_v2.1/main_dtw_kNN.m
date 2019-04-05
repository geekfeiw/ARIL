
clear; close all; clc;
mex dtw_c.c;
load('../../data/train_data_split_amp.mat');
load('../../data/test_data_split_amp.mat');

% for i = 1:size(train_data,1)
%     temp = squeeze(train_data(i,:,:));
%     train_data_reshape(i,:) = temp(:); 
% end
%     
% for i = 1:size(test_data,1)
%     temp = squeeze(test_data(i,:,:));
%     test_data_reshape(i,:) = temp(:); 
% end

subcarrier_num = 52;
train_size = size(train_data, 1);
test_size = size(test_data, 1);
w = 50;

prediction_loction = zeros([test_size, 1]);
prediction_activity = zeros([test_size, 1]);
tic;
for i = 1: test_size 
    i
    distance_onetestsample_trainset = zeros([train_size,1]);
    for j = 1:train_size
        
        temp_distance = 0;
        for k = 1:subcarrier_num   
            a = squeeze(test_data(i,k,:));
            b = squeeze(train_data(j,k,:));
            d = dtw_c(a,b,w);
            % sum the distances that between all subcarriers
            temp_distance = temp_distance + d;
        end
        
        % the (i-test, j_train) distance
        distance_onetestsample_trainset(j) = temp_distance;       
       
    end
    
    % knn, k=1
    [~, idx] = min(distance_onetestsample_trainset);    
    prediction_loction(i) = train_location_label(idx);
    prediction_activity(i) = train_activity_label(idx);
    
end
 toc;       
location_accuracy = sum(prediction_loction==test_location_label)/test_size;
activity_accuracy = sum(prediction_activity==test_activity_label)/test_size;

  
  
  
  
  
        