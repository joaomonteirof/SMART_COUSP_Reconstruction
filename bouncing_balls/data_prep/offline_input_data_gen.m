%% forward model of COSUP system
Nor = @(x) ( x-min(x(:)) )/( max(x(:)) - min(x(:)) );

%% load video set and mask
load('Raw_video_AI_CUI_256x256_100frames_4samples.mat') %% video set 
load 'experiment_mask.mat' %% experiment mask (gray mask)
load 'simulation mask.mat' %% simulation mask (binary mask)


figure
subplot(121) 
imshow(d);title('simulation mask');
subplot(122)
imshow(mask_);title('experiment mask');


im = permute(Data, [2,3,4, 1]); %% video set 
im(:,:,1:3,:) = 0; %% actually I used 97 frames and zero the first 3 frames

%% Cu 
for i = 1:100
    Cu(:,1+(i-1):(i-1)+256,i) = mask_; %% experiment mask
end

%% shearing video x

foo = im(:,:,:,1);
x = zeros( size(Cu));
for i = 1:100
    bar = foo(:,:,100-i+1);
    tform = projective2d([1.0079 0.0085 0.0001;0.0226 1.0155 0.0001;0.9163 0.6183 1.0000]);
    bar_T = imwarp(bar, tform, 'OutputView', imref2d( size(bar) ));   
    x(:,1 + (i-1):(i-1)+256,i) = bar_T; 
end

%% y1 streak image

y1 = Nor( sum( Cu.*x, 3 ) );
figure
imshow(y1)
% imwrite(y1, 'streak_simulation_data_20sa21.tif')