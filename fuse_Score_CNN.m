%     clc;
clear;

u = load('Score_infrared_diff_frames16x4_mvit_v2_s_one_fc512_bs8_rnn.mat');
v =  load('Score_infrared_frames16x4_mvit_v2_s_one_fc512_bs8_rnn.mat');


x = load('Score_infrared16x4_mvit_mhsa.mat');
y =    load('Score_color16x4_mvit_mhsa.mat');

z =  load('e4r5.mat');

ds=       load('infrared_rgbmm1_test.mat');

ctr = 0;

testSize = size(ds.y2,2);
numClasses = max(ds.y2);
confMat = zeros(numClasses, numClasses);

for i=1:testSize
    scores = x.y_pred(i,:);
    scores = y.y_pred(i,:);
    scores = z.y_pred(i,:);

    [bestScore, best] = max(scores);
    if ds.y2(1,i)==best
        ctr = ctr+1;
    end
    confMat(ds.y2(1,i),best) = confMat(ds.y2(1,i),best)+1;
end

accuracy = ctr/(testSize)*100

