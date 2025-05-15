clc
clear;



fn = 'train/infrared/wave2/000063.avi';
% fn = 'train/color/wave2/000063.avi';

vidReader = VideoReader(fn);
nof = vidReader.NumberOfFrames;
step = floor(nof/50);
for i=1:1:nof
%     im = read(vidReader,i+1) - read(vidReader,i);
    im = read(vidReader,i);
%     im = im(40:439,:,:);
%     im = imresize(im, 0.64);
%     im = im(:,84:376,:);
    imwrite(im, strcat('frames/', int2str(i), '.jpg'));
end





