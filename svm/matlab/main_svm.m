
clear
load('../../data/train_data_split_amp.mat');
load('../../data/test_data_split_amp.mat');



train_data_reshape = zeros([length(train_activity_label),52*192]);
test_data_reshape = zeros([length(test_activity_label),52*192]);


for i = 1:length(train_activity_label)
    temp = squeeze(train_data(i,:,:));
    train_data_reshape(i,:) = temp(:);
end

for i = 1:length(test_activity_label)
    temp = squeeze(test_data(i,:,:));
    test_data_reshape(i,:) = temp(:);
end


train_data_reshape = train_data_reshape';
[train_data_norm, pattern] = mapminmax(train_data_reshape,-1,1);
train_data_norm = train_data_norm';

test_data_reshape = test_data_reshape';
test_data_norm = mapminmax('apply',test_data_reshape,pattern);
test_data_norm = test_data_norm';

tic

model_act = svmtrain(train_activity_label, double(train_data_norm),'-s 0 -t 2');
[predict_label_act, a]=svmpredict(test_activity_label, double(test_data_norm), model_act);
model_loc = svmtrain(train_location_label, double(train_data_norm),'-s 0 -t 2');
[predict_label_loc, b]=svmpredict(test_location_label, double(test_data_norm), model_loc);

toc