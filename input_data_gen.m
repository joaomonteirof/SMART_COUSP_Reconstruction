clear all;
clc;

%% Xianglei updated it on 10/22/2019. This script is used for the forward model of CUP (i.e. generate encoded, shifted (Along horizontal direction i.e. column), and integrated streak image)
%%

% define function handle: Normalization
Nor = @(x) (x - min(x(:)))/(max(x(:)) - min(x(:)));

video_samples_file = 'data/targets/output_train.hdf';
output_file_name = 'data/input/input_train.hdf';


%% Load bouncing balls data
hinfo = hdf5info(video_samples_file);
Data = hdf5read(hinfo.GroupHierarchy.Datasets(1));
Data=permute(Data,[4, 3, 2, 1]); % xianglei changed it from [4, 2, 3, 1]

%% Set N: Frames dim1, Nx: Frames dim2, Nt: #frames, Np: ... 
[N, D_x,D_y,D_t]=size(Data);

%Rm=randn(D_x, D_y);      
mask=load('mask.mat');  % xianglei updated a new experimental mask
Rm=mask.mask2;
C=Rm;

%% Loop over video samples to generate a list of streak images and patterns
y_all = {};
for k=1:N
    x=Data(k,:,:,:);
    x=squeeze(x);
    
	%p=4;     % Xianglei updated this part. Since I have finished the binning operator in the experiment, the mask we used here does not need this operator anymore.
	%C_transiton=zeros(p*size(C,1),p*size(C,2));
	%for i=1:size(C,1)
	    %for j=1:size(C,2)
            %C_transiton(1+p*(i-1):p*i,1+p*(j-1):p*j)=C(i,j);
	    %end
	%end
	%C=im2bw(C_transiton(1:D_x,1:D_y),0.1);
	%figure;imshow(C,[]); 

	C_1=zeros(D_x,D_y+D_t-1,D_t);      % shifting the encoding mask C along the horizontal direction         
	for i=1:D_t            
	    C_1(:,i:i-1+D_y,i)=C;                   % xianglei changed from i:i-1+D_x,:,i
	    figure(10);imshow(C_1(:,:,i),[]);  
	end
	Cu=C_1;

	x_1=zeros(D_x,D_y+D_t-1,D_t);      % shifting the video sample  along the horizontal directionthe            
	for i=1:D_t            
	    x_1(:,i:i-1+D_y,i)=x(:,:,i);             % xianglei changed from i:i-1+D_x,:,i
	    figure(10);imshow(x_1(:,:,i),[]);  
    end
    
	x=x_1;
	y1=x.*Cu;                        % encoding the video with the shearing operator
	y = sum(y1,3);                   % integrate to obtain a streaking image

    %% Store
    y_all{k} = y;
    
end

y = cat(3,y_all{:});
y = permute(y,[3,2,1]);
%mean_y = mean(reshape(y,[numel(y),1]));  % Since we do simulation, I think we do not need this post-process.
%std_y = std(reshape(y,[numel(y),1]));
%y = (y-mean_y)/std_y;
y = Nor(y);

%% Save y
h5create(output_file_name,'/input_samples',size(y))
h5write(output_file_name,'/input_samples',y);
